from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.normalization.normalizer import EFTNormalizer


NAME_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9])"
    r"(?:[A-Za-z]{0,3}\d{6,}|\d{6,}[A-Za-z]{0,3})"
    r"(?![A-Za-z0-9])",
    flags=re.IGNORECASE,
)


def extract_name_identifiers(
    text: str,
) -> set[str]:
    """
    Şirket adındaki uzun sayısal veya alfanümerik
    tanımlayıcıları çıkarır.

    Örnek:
        Custody Bank /041159338/933802
        -> {"041159338", "933802"}
    """

    if not text:
        return set()

    return {
        match.group(0).upper()
        for match in NAME_IDENTIFIER_PATTERN.finditer(
            text
        )
    }


def calculate_jaro_winkler_score(
    first_text: str,
    second_text: str,
) -> float:
    """
    İki metin arasındaki normalize Jaro-Winkler
    benzerliğini 0 ile 1 arasında hesaplar.
    """

    if not first_text or not second_text:
        return 0.0

    score = JaroWinkler.normalized_similarity(
        first_text,
        second_text,
        prefix_weight=0.1,
        processor=None,
    )

    return max(
        0.0,
        min(
            1.0,
            float(score),
        ),
    )


def calculate_single_token_prefix_score(
    query_token: str,
    candidate_token: str,
) -> float:
    """
    Tek bir sorgu tokenının aday tokenın kısaltılmış
    başlangıcı olup olmadığını ölçer.

    Örnek:
        HOLDI -> HOLDINGS
        ENERJ -> ENERJISA
    """

    if not query_token or not candidate_token:
        return 0.0

    if query_token == candidate_token:
        return 1.0

    shorter_token = (
        query_token
        if len(query_token)
        <= len(candidate_token)
        else candidate_token
    )

    longer_token = (
        candidate_token
        if len(candidate_token)
        >= len(query_token)
        else query_token
    )

    # İki karakterli veya daha kısa prefixler
    # fazla sayıda yanlış eşleşme üretebilir.
    if len(shorter_token) < 3:
        return 0.0

    if longer_token.startswith(
        shorter_token
    ):
        length_coverage = (
            len(shorter_token)
            / len(longer_token)
        )

        return min(
            1.0,
            0.70
            + 0.30
            * length_coverage,
        )

    common_prefix_length = 0

    for (
        query_character,
        candidate_character,
    ) in zip(
        query_token,
        candidate_token,
    ):
        if (
            query_character
            != candidate_character
        ):
            break

        common_prefix_length += 1

    if common_prefix_length < 2:
        return 0.0

    common_prefix_ratio = (
        common_prefix_length
        / max(
            len(query_token),
            len(candidate_token),
        )
    )

    # Tam prefix olmayan kısmi başlangıç
    # benzerliği daha düşük ağırlık alır.
    return min(
        0.55,
        0.55
        * common_prefix_ratio,
    )


def calculate_token_prefix_score(
    query_text: str,
    candidate_text: str,
) -> float:
    """
    Sorgudaki her tokenı adayın en uygun tokenı
    ile karşılaştırır.

    Sorgu şirket adının kısaltılmış hâli olabileceği
    için değerlendirme sorgu tokenları üzerinden yapılır.
    """

    query_tokens = [
        token
        for token in query_text.split()
        if token
    ]

    candidate_tokens = [
        token
        for token in candidate_text.split()
        if token
    ]

    if not query_tokens or not candidate_tokens:
        return 0.0

    weighted_score_sum = 0.0
    weight_sum = 0.0

    for query_token in query_tokens:
        best_token_score = max(
            calculate_single_token_prefix_score(
                query_token,
                candidate_token,
            )
            for candidate_token
            in candidate_tokens
        )

        # Daha uzun tokenlar şirketi ayırt etmede
        # genellikle daha fazla bilgi taşır.
        token_weight = float(
            max(
                1,
                min(
                    len(query_token),
                    8,
                ),
            )
        )

        weighted_score_sum += (
            best_token_score
            * token_weight
        )

        weight_sum += token_weight

    if weight_sum <= 0.0:
        return 0.0

    score = (
        weighted_score_sum
        / weight_sum
    )

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


class EmbeddingProvider(Protocol):
    is_loaded: bool
    embedding_dimension: int

    def encode_query(
        self,
        text: str,
    ) -> np.ndarray:
        ...

    def encode_documents(
        self,
        texts: list[str],
    ) -> np.ndarray:
        ...


@dataclass(frozen=True)
class CompanyRecord:
    company_id: int
    legal_name: str
    brand_name: str
    city: str
    sector: str
    tax_number: str


@dataclass(frozen=True)
class AliasRecord:
    alias_id: int
    company_id: int
    alias_text: str
    normalized_alias: str
    alias_type: str


class CompanyIndex:
    """
    Yerel hibrit şirket arama indeksi.

    Arama kanalları:
    - Exact alias
    - Character-word-boundary TF-IDF
    - Raw-character TF-IDF
    - RapidFuzz
    - Jaro-Winkler
    - Token-prefix similarity
    - Sayısal tanımlayıcı eşleşmesi
    - Qwen3 semantic embedding
    """

    def __init__(
        self,
        companies: dict[int, CompanyRecord],
        aliases: list[AliasRecord],
        normalizer: EFTNormalizer | None = None,
    ) -> None:
        if not companies:
            raise ValueError(
                "Şirket veri seti boş olamaz."
            )

        if not aliases:
            raise ValueError(
                "Alias veri seti boş olamaz."
            )

        self.companies = companies
        self.aliases = aliases

        self.normalizer = (
            normalizer
            or EFTNormalizer()
        )

        self.alias_texts = [
            alias.normalized_alias
            for alias in self.aliases
        ]

        self.alias_identifier_sets = [
            extract_name_identifiers(
                alias.alias_text
            )
            for alias in self.aliases
        ]

        self.exact_alias_map = (
            self._build_exact_alias_map()
        )

        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 5),
            lowercase=False,
            norm="l2",
            sublinear_tf=True,
            dtype=np.float32,
        )

        self.alias_matrix = (
            self.vectorizer.fit_transform(
                self.alias_texts
            )
        )

        self.raw_char_vectorizer = (
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(2, 5),
                lowercase=False,
                norm="l2",
                sublinear_tf=True,
                dtype=np.float32,
            )
        )

        self.raw_char_alias_matrix = (
            self.raw_char_vectorizer.fit_transform(
                self.alias_texts
            )
        )

        self.embedding_model: (
            EmbeddingProvider | None
        ) = None

        self.embedding_documents: list[str] = []

        self.alias_embedding_matrix: (
            np.ndarray | None
        ) = None

    @classmethod
    def from_csv(
        cls,
        companies_path: str | Path,
        aliases_path: str | Path,
    ) -> "CompanyIndex":
        companies_path = Path(
            companies_path
        )

        aliases_path = Path(
            aliases_path
        )

        if not companies_path.exists():
            raise FileNotFoundError(
                "Şirket dosyası bulunamadı: "
                f"{companies_path}"
            )

        if not aliases_path.exists():
            raise FileNotFoundError(
                "Alias dosyası bulunamadı: "
                f"{aliases_path}"
            )

        normalizer = EFTNormalizer()

        companies = cls._read_companies(
            companies_path
        )

        explicit_aliases = cls._read_aliases(
            aliases_path=aliases_path,
            companies=companies,
            normalizer=normalizer,
        )

        aliases = (
            cls._add_implicit_company_aliases(
                companies=companies,
                aliases=explicit_aliases,
                normalizer=normalizer,
            )
        )

        return cls(
            companies=companies,
            aliases=aliases,
            normalizer=normalizer,
        )

    def attach_embedding_model(
        self,
        embedding_model: EmbeddingProvider,
    ) -> None:
        if not embedding_model.is_loaded:
            raise RuntimeError(
                "Embedding modeli yüklenmeden "
                "indekse bağlanamaz."
            )

        documents = [
            self._build_embedding_document(
                alias
            )
            for alias in self.aliases
        ]

        embedding_matrix = (
            embedding_model.encode_documents(
                documents
            )
        )

        if embedding_matrix.ndim != 2:
            raise RuntimeError(
                "Alias embedding matrisi "
                "iki boyutlu değil."
            )

        if (
            embedding_matrix.shape[0]
            != len(self.aliases)
        ):
            raise RuntimeError(
                "Embedding satır sayısı alias "
                "sayısıyla eşleşmiyor."
            )

        self.embedding_model = (
            embedding_model
        )

        self.embedding_documents = (
            documents
        )

        self.alias_embedding_matrix = (
            embedding_matrix
        )

    def detach_embedding_model(
        self,
    ) -> None:
        self.embedding_model = None
        self.embedding_documents = []
        self.alias_embedding_matrix = None

    @staticmethod
    def _read_companies(
        path: Path,
    ) -> dict[int, CompanyRecord]:
        companies: dict[
            int,
            CompanyRecord,
        ] = {}

        with path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file
            )

            required_columns = {
                "company_id",
                "legal_name",
                "brand_name",
                "city",
                "sector",
                "tax_number",
            }

            missing_columns = (
                required_columns.difference(
                    reader.fieldnames or []
                )
            )

            if missing_columns:
                raise ValueError(
                    "companies.csv eksik "
                    "kolon içeriyor: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                company_id = int(
                    row["company_id"]
                )

                if company_id in companies:
                    raise ValueError(
                        "Tekrarlanan company_id: "
                        f"{company_id}"
                    )

                companies[
                    company_id
                ] = CompanyRecord(
                    company_id=company_id,
                    legal_name=(
                        row[
                            "legal_name"
                        ].strip()
                    ),
                    brand_name=(
                        row[
                            "brand_name"
                        ].strip()
                    ),
                    city=(
                        row[
                            "city"
                        ].strip()
                    ),
                    sector=(
                        row[
                            "sector"
                        ].strip()
                    ),
                    tax_number=(
                        row[
                            "tax_number"
                        ].strip()
                    ),
                )

        return companies

    @staticmethod
    def _read_aliases(
        aliases_path: Path,
        companies: dict[int, CompanyRecord],
        normalizer: EFTNormalizer,
    ) -> list[AliasRecord]:
        aliases: list[
            AliasRecord
        ] = []

        with aliases_path.open(
            mode="r",
            encoding="utf-8-sig",
            newline="",
        ) as csv_file:
            reader = csv.DictReader(
                csv_file
            )

            required_columns = {
                "alias_id",
                "company_id",
                "alias_text",
                "alias_type",
            }

            missing_columns = (
                required_columns.difference(
                    reader.fieldnames or []
                )
            )

            if missing_columns:
                raise ValueError(
                    "aliases.csv eksik kolon "
                    "içeriyor: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                alias_id = int(
                    row["alias_id"]
                )

                company_id = int(
                    row["company_id"]
                )

                alias_text = (
                    row[
                        "alias_text"
                    ].strip()
                )

                if company_id not in companies:
                    raise ValueError(
                        "Alias bilinmeyen şirkete "
                        "bağlı: "
                        f"company_id={company_id}"
                    )

                normalized_alias = (
                    CompanyIndex._normalize_alias(
                        alias_text,
                        normalizer,
                    )
                )

                if not normalized_alias:
                    continue

                aliases.append(
                    AliasRecord(
                        alias_id=alias_id,
                        company_id=company_id,
                        alias_text=alias_text,
                        normalized_alias=(
                            normalized_alias
                        ),
                        alias_type=(
                            row[
                                "alias_type"
                            ].strip()
                        ),
                    )
                )

        return aliases

    @staticmethod
    def _add_implicit_company_aliases(
        companies: dict[int, CompanyRecord],
        aliases: list[AliasRecord],
        normalizer: EFTNormalizer,
    ) -> list[AliasRecord]:
        result = list(
            aliases
        )

        existing_aliases = {
            (
                alias.company_id,
                alias.normalized_alias,
            )
            for alias in result
        }

        generated_alias_id = -1

        for company in companies.values():
            possible_aliases = (
                (
                    "LEGAL_NAME",
                    company.legal_name,
                ),
                (
                    "BRAND_NAME",
                    company.brand_name,
                ),
            )

            for (
                alias_type,
                alias_text,
            ) in possible_aliases:
                if not alias_text:
                    continue

                normalized_alias = (
                    CompanyIndex._normalize_alias(
                        alias_text,
                        normalizer,
                    )
                )

                identity = (
                    company.company_id,
                    normalized_alias,
                )

                if not normalized_alias:
                    continue

                if (
                    identity
                    in existing_aliases
                ):
                    continue

                result.append(
                    AliasRecord(
                        alias_id=(
                            generated_alias_id
                        ),
                        company_id=(
                            company.company_id
                        ),
                        alias_text=alias_text,
                        normalized_alias=(
                            normalized_alias
                        ),
                        alias_type=(
                            alias_type
                        ),
                    )
                )

                existing_aliases.add(
                    identity
                )

                generated_alias_id -= 1

        return result

    @staticmethod
    def _normalize_alias(
        alias_text: str,
        normalizer: EFTNormalizer,
    ) -> str:
        result = normalizer.normalize(
            alias_text
        )

        return (
            result["core_text"]
            or result["ascii_text"]
        ).strip()

    def _build_exact_alias_map(
        self,
    ) -> dict[str, list[int]]:
        exact_alias_map: dict[
            str,
            list[int],
        ] = {}

        for index, alias in enumerate(
            self.aliases
        ):
            exact_alias_map.setdefault(
                alias.normalized_alias,
                [],
            ).append(
                index
            )

        return exact_alias_map

    def _build_embedding_document(
        self,
        alias: AliasRecord,
    ) -> str:
        company = self.companies[
            alias.company_id
        ]

        return (
            "Official company name: "
            f"{company.legal_name} | "
            "Brand: "
            f"{company.brand_name} | "
            "Known alias: "
            f"{alias.alias_text} | "
            "Normalized alias: "
            f"{alias.normalized_alias} | "
            "City: "
            f"{company.city} | "
            "Sector: "
            f"{company.sector}"
        )

    @staticmethod
    def _is_safe_exact_query(
        query: str,
    ) -> bool:
        alphanumeric_length = sum(
            character.isalnum()
            for character in query
        )

        return (
            alphanumeric_length
            >= 3
        )

    def search(
        self,
        text: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        if limit < 1:
            raise ValueError(
                "limit en az 1 olmalıdır."
            )

        normalization = (
            self.normalizer.normalize(
                text
            )
        )

        query = (
            normalization["core_text"]
            or normalization["ascii_text"]
        ).strip()

        if not query:
            return self._empty_result(
                normalization
            )

        query_identifiers = (
            extract_name_identifiers(
                text
            )
        )

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        char_wb_tfidf_scores = (
            cosine_similarity(
                query_vector,
                self.alias_matrix,
            ).ravel()
        )

        raw_char_query_vector = (
            self.raw_char_vectorizer.transform(
                [query]
            )
        )

        raw_char_tfidf_scores = (
            cosine_similarity(
                raw_char_query_vector,
                self.raw_char_alias_matrix,
            ).ravel()
        )

        combined_char_tfidf_scores = (
            0.55
            * char_wb_tfidf_scores
            + 0.45
            * raw_char_tfidf_scores
        )

        jaro_winkler_scores = np.fromiter(
            (
                calculate_jaro_winkler_score(
                    query,
                    alias_text,
                )
                for alias_text
                in self.alias_texts
            ),
            dtype=np.float32,
            count=len(
                self.alias_texts
            ),
        )

        retrieval_limit = min(
            len(self.aliases),
            max(
                limit * 10,
                50,
            ),
        )

        candidate_indices: set[
            int
        ] = set()

        exact_indices = (
            self.exact_alias_map.get(
                query,
                [],
            )
        )

        candidate_indices.update(
            exact_indices
        )

        exact_company_ids = {
            self.aliases[
                index
            ].company_id
            for index
            in exact_indices
        }

        unique_exact_company_id = (
            next(
                iter(
                    exact_company_ids
                )
            )
            if len(
                exact_company_ids
            ) == 1
            else None
        )

        safe_exact_query = (
            self._is_safe_exact_query(
                query
            )
        )

        char_wb_top_indices = np.argsort(
            -char_wb_tfidf_scores
        )[:retrieval_limit]

        candidate_indices.update(
            int(index)
            for index
            in char_wb_top_indices
        )

        raw_char_top_indices = np.argsort(
            -raw_char_tfidf_scores
        )[:retrieval_limit]

        candidate_indices.update(
            int(index)
            for index
            in raw_char_top_indices
        )

        combined_char_top_indices = np.argsort(
            -combined_char_tfidf_scores
        )[:retrieval_limit]

        candidate_indices.update(
            int(index)
            for index
            in combined_char_top_indices
        )

        jaro_winkler_top_indices = np.argsort(
            -jaro_winkler_scores
        )[:retrieval_limit]

        candidate_indices.update(
            int(index)
            for index
            in jaro_winkler_top_indices
        )

        fuzzy_results = process.extract(
            query,
            self.alias_texts,
            scorer=fuzz.WRatio,
            processor=None,
            limit=retrieval_limit,
            score_cutoff=0,
        )

        candidate_indices.update(
            int(result[2])
            for result
            in fuzzy_results
        )

        if query_identifiers:
            for (
                alias_index,
                identifiers,
            ) in enumerate(
                self.alias_identifier_sets
            ):
                if (
                    query_identifiers
                    & identifiers
                ):
                    candidate_indices.add(
                        alias_index
                    )

        embedding_enabled = (
            self.embedding_model
            is not None
            and self.alias_embedding_matrix
            is not None
        )

        embedding_scores = np.zeros(
            len(self.aliases),
            dtype=np.float32,
        )

        if embedding_enabled:
            query_embedding = (
                self.embedding_model.encode_query(
                    query
                )
            )

            embedding_scores = (
                self.alias_embedding_matrix
                @ query_embedding
            )

            embedding_top_indices = np.argsort(
                -embedding_scores
            )[:retrieval_limit]

            candidate_indices.update(
                int(index)
                for index
                in embedding_top_indices
            )

        company_candidates: dict[
            int,
            dict[str, Any],
        ] = {}

        for alias_index in candidate_indices:
            alias = self.aliases[
                alias_index
            ]

            company = self.companies[
                alias.company_id
            ]

            exact_match = (
                query
                == alias.normalized_alias
            )

            unique_exact_match = (
                exact_match
                and safe_exact_query
                and unique_exact_company_id
                == alias.company_id
            )

            ambiguous_exact_match = (
                exact_match
                and not unique_exact_match
            )

            alias_identifiers = (
                self.alias_identifier_sets[
                    alias_index
                ]
            )

            identifier_overlap = bool(
                query_identifiers
                and alias_identifiers
                and (
                    query_identifiers
                    & alias_identifiers
                )
            )

            identifier_score = (
                1.0
                if identifier_overlap
                else 0.0
            )

            char_wb_tfidf_score = float(
                char_wb_tfidf_scores[
                    alias_index
                ]
            )

            raw_char_tfidf_score = float(
                raw_char_tfidf_scores[
                    alias_index
                ]
            )

            char_tfidf_score = float(
                combined_char_tfidf_scores[
                    alias_index
                ]
            )

            jaro_winkler_score = float(
                jaro_winkler_scores[
                    alias_index
                ]
            )

            token_prefix_score = (
                calculate_token_prefix_score(
                    query,
                    alias.normalized_alias,
                )
            )

            fuzzy_ratio_score = (
                fuzz.ratio(
                    query,
                    alias.normalized_alias,
                )
                / 100.0
            )

            fuzzy_partial_ratio_score = (
                fuzz.partial_ratio(
                    query,
                    alias.normalized_alias,
                )
                / 100.0
            )

            fuzzy_token_set_score = (
                fuzz.token_set_ratio(
                    query,
                    alias.normalized_alias,
                )
                / 100.0
            )

            fuzzy_wratio_score = (
                fuzz.WRatio(
                    query,
                    alias.normalized_alias,
                )
                / 100.0
            )

            raw_embedding_score = float(
                embedding_scores[
                    alias_index
                ]
            )

            embedding_score = max(
                0.0,
                min(
                    1.0,
                    raw_embedding_score,
                ),
            )

            if unique_exact_match:
                hybrid_score = 1.0

            elif embedding_enabled:
                hybrid_score = (
                    0.20
                    * char_tfidf_score
                    + 0.15
                    * fuzzy_wratio_score
                    + 0.08
                    * fuzzy_partial_ratio_score
                    + 0.07
                    * fuzzy_token_set_score
                    + 0.12
                    * jaro_winkler_score
                    + 0.08
                    * token_prefix_score
                    + 0.20
                    * embedding_score
                    + 0.10
                    * identifier_score
                )

            else:
                hybrid_score = (
                    0.30
                    * char_tfidf_score
                    + 0.20
                    * fuzzy_wratio_score
                    + 0.10
                    * fuzzy_partial_ratio_score
                    + 0.10
                    * fuzzy_token_set_score
                    + 0.10
                    * jaro_winkler_score
                    + 0.10
                    * token_prefix_score
                    + 0.10
                    * identifier_score
                )

            hybrid_score = max(
                0.0,
                min(
                    1.0,
                    hybrid_score,
                ),
            )

            candidate = {
                "company_id": (
                    company.company_id
                ),
                "legal_name": (
                    company.legal_name
                ),
                "brand_name": (
                    company.brand_name
                ),
                "city": (
                    company.city
                ),
                "sector": (
                    company.sector
                ),
                "matched_alias": (
                    alias.alias_text
                ),
                "normalized_alias": (
                    alias.normalized_alias
                ),
                "alias_type": (
                    alias.alias_type
                ),
                "exact_match": (
                    exact_match
                ),
                "unique_exact_match": (
                    unique_exact_match
                ),
                "ambiguous_exact_match": (
                    ambiguous_exact_match
                ),
                "identifier_score": round(
                    identifier_score,
                    6,
                ),
                "identifier_overlap": (
                    identifier_overlap
                ),
                "char_wb_tfidf_score": round(
                    char_wb_tfidf_score,
                    6,
                ),
                "raw_char_tfidf_score": round(
                    raw_char_tfidf_score,
                    6,
                ),
                "char_tfidf_score": round(
                    char_tfidf_score,
                    6,
                ),
                "jaro_winkler_score": round(
                    jaro_winkler_score,
                    6,
                ),
                "token_prefix_score": round(
                    token_prefix_score,
                    6,
                ),
                "fuzzy_ratio_score": round(
                    fuzzy_ratio_score,
                    6,
                ),
                "fuzzy_partial_ratio_score": (
                    round(
                        fuzzy_partial_ratio_score,
                        6,
                    )
                ),
                "fuzzy_token_set_score": (
                    round(
                        fuzzy_token_set_score,
                        6,
                    )
                ),
                "fuzzy_wratio_score": round(
                    fuzzy_wratio_score,
                    6,
                ),
                "embedding_score": round(
                    embedding_score,
                    6,
                ),
                "hybrid_score": round(
                    hybrid_score,
                    6,
                ),
            }

            current_candidate = (
                company_candidates.get(
                    company.company_id
                )
            )

            if (
                current_candidate
                is None
                or candidate[
                    "hybrid_score"
                ]
                > current_candidate[
                    "hybrid_score"
                ]
            ):
                company_candidates[
                    company.company_id
                ] = candidate

        ranked_candidates = sorted(
            company_candidates.values(),
            key=lambda candidate: (
                bool(
                    candidate[
                        "unique_exact_match"
                    ]
                ),
                candidate[
                    "hybrid_score"
                ],
            ),
            reverse=True,
        )[:limit]

        decision, margin = (
            self._make_prototype_decision(
                ranked_candidates
            )
        )

        return {
            "normalization": (
                normalization
            ),
            "query_text": (
                query
            ),
            "decision": (
                decision
            ),
            "best_candidate": (
                ranked_candidates[0]
                if ranked_candidates
                else None
            ),
            "score_margin": (
                margin
            ),
            "candidates": (
                ranked_candidates
            ),
            "index_stats": (
                self.get_stats()
            ),
        }

    @staticmethod
    def _make_prototype_decision(
        candidates: list[
            dict[str, Any]
        ],
    ) -> tuple[str, float]:
        if not candidates:
            return (
                "UNKNOWN",
                0.0,
            )

        best_candidate = (
            candidates[0]
        )

        best_score = float(
            best_candidate[
                "hybrid_score"
            ]
        )

        second_score = (
            float(
                candidates[1][
                    "hybrid_score"
                ]
            )
            if len(candidates) > 1
            else 0.0
        )

        margin = round(
            best_score
            - second_score,
            6,
        )

        if best_candidate.get(
            "unique_exact_match",
            False,
        ):
            return (
                "AUTO_MATCH",
                margin,
            )

        if best_candidate.get(
            "ambiguous_exact_match",
            False,
        ):
            return (
                "MANUAL_REVIEW",
                margin,
            )

        if (
            best_score >= 0.90
            and margin >= 0.08
        ):
            return (
                "AUTO_MATCH",
                margin,
            )

        if best_score >= 0.58:
            return (
                "MANUAL_REVIEW",
                margin,
            )

        return (
            "UNKNOWN",
            margin,
        )

    def get_stats(
        self,
    ) -> dict[str, Any]:
        embedding_enabled = (
            self.embedding_model
            is not None
            and self.alias_embedding_matrix
            is not None
        )

        ambiguous_exact_alias_count = sum(
            1
            for indices
            in self.exact_alias_map.values()
            if len(
                {
                    self.aliases[
                        index
                    ].company_id
                    for index
                    in indices
                }
            ) > 1
        )

        return {
            "company_count": len(
                self.companies
            ),
            "alias_count": len(
                self.aliases
            ),
            "tfidf_feature_count": len(
                self.vectorizer.vocabulary_
            ),
            "char_wb_tfidf_feature_count": (
                len(
                    self.vectorizer.vocabulary_
                )
            ),
            "raw_char_tfidf_feature_count": (
                len(
                    self.raw_char_vectorizer
                    .vocabulary_
                )
            ),
            "raw_char_tfidf_enabled": True,
            "jaro_winkler_enabled": True,
            "token_prefix_enabled": True,
            "ambiguous_exact_alias_count": (
                ambiguous_exact_alias_count
            ),
            "embedding_enabled": (
                embedding_enabled
            ),
            "embedding_dimension": (
                int(
                    self.alias_embedding_matrix
                    .shape[1]
                )
                if embedding_enabled
                else 0
            ),
        }

    def _empty_result(
        self,
        normalization: dict[
            str,
            Any,
        ],
    ) -> dict[str, Any]:
        return {
            "normalization": (
                normalization
            ),
            "query_text": "",
            "decision": "UNKNOWN",
            "best_candidate": None,
            "score_margin": 0.0,
            "candidates": [],
            "index_stats": (
                self.get_stats()
            ),
        }