from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

BACKEND_ROOT = (
    PROJECT_ROOT
    / "backend"
)

if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(BACKEND_ROOT),
    )


from app.normalization.normalizer import EFTNormalizer


DEFAULT_COMPANIES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "companies.csv"
)

DEFAULT_ALIASES = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aliases.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "hard_positive_benchmark_queries.csv"
)


TOKEN_RE = re.compile(
    r"[^\W_]+",
    re.UNICODE,
)

SPACE_RE = re.compile(
    r"\s+"
)


LEGAL_TOKENS = {
    "A",
    "AS",
    "AŞ",
    "ANONIM",
    "ANONİM",
    "SIRKET",
    "ŞİRKET",
    "SIRKETI",
    "ŞİRKETİ",
    "LIMITED",
    "LİMİTED",
    "LTD",
    "LLC",
    "LP",
    "INC",
    "CORP",
    "CORPORATION",
    "COMPANY",
    "CO",
    "GMBH",
    "BV",
    "NV",
    "SAS",
    "SA",
    "PLC",
    "HOLDING",
    "HOLDİNG",
}


VOWELS = set(
    "AEIİOÖUÜ"
    "aeıioöuü"
)


NOISE = (
    "ODM",
    "ODEME",
    "FATURA ODEMESI",
    "HAVALE",
    "EFT",
    "MAL BEDELI",
    "HIZMET BEDELI",
    "SIPARIS ODEMESI",
)


SUBSTITUTIONS = {
    "A": "E",
    "E": "A",
    "I": "İ",
    "İ": "I",
    "O": "U",
    "U": "O",
    "S": "Ş",
    "Ş": "S",
    "C": "Ç",
    "Ç": "C",
    "G": "Ğ",
    "Ğ": "G",
    "K": "G",
    "T": "D",
    "B": "P",
    "P": "B",
}


normalizer = EFTNormalizer()


@dataclass(frozen=True)
class SourceAlias:
    company_id: str
    legal_name: str
    alias_text: str
    alias_type: str
    normalized_alias: str


@dataclass(frozen=True)
class Pipeline:
    name: str
    operations: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Doğru şirketin mevcut sıralamada 2-10 "
            "arasında bulunduğu hard-positive "
            "benchmark sorguları üretir."
        )
    )

    parser.add_argument(
        "--companies-path",
        type=Path,
        default=DEFAULT_COMPANIES,
    )

    parser.add_argument(
        "--aliases-path",
        type=Path,
        default=DEFAULT_ALIASES,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--backend-url",
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--minimum-expected-rank",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--maximum-expected-rank",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--max-attempts",
        type=int,
        default=20000,
    )

    parser.add_argument(
        "--max-per-company",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
) -> None:
    if args.target_count < 1:
        raise ValueError(
            "--target-count en az 1 olmalıdır."
        )

    if args.candidate_limit < 2:
        raise ValueError(
            "--candidate-limit en az 2 olmalıdır."
        )

    if args.minimum_expected_rank < 2:
        raise ValueError(
            "--minimum-expected-rank en az 2 olmalıdır."
        )

    if (
        args.maximum_expected_rank
        < args.minimum_expected_rank
    ):
        raise ValueError(
            "Maksimum sıra minimum sıradan "
            "küçük olamaz."
        )

    if (
        args.maximum_expected_rank
        > args.candidate_limit
    ):
        raise ValueError(
            "Maksimum sıra candidate-limit "
            "değerini aşamaz."
        )

    if (
        args.max_attempts
        < args.target_count
    ):
        raise ValueError(
            "--max-attempts target-count "
            "değerinden küçük olamaz."
        )

    if args.max_per_company < 1:
        raise ValueError(
            "--max-per-company en az 1 olmalıdır."
        )

    if args.timeout <= 0:
        raise ValueError(
            "--timeout pozitif olmalıdır."
        )


def read_csv(
    path: Path,
    label: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{label} bulunamadı: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(
            csv.DictReader(handle)
        )

    if not rows:
        raise ValueError(
            f"{label} boş."
        )

    return rows


def clean(
    text: str,
) -> str:
    return SPACE_RE.sub(
        " ",
        text,
    ).strip()


def normalize(
    text: str,
) -> str:
    result = normalizer.normalize(
        text
    )

    return (
        result.get("core_text")
        or result.get("ascii_text")
        or ""
    ).strip()


def tokens(
    text: str,
) -> list[str]:
    return TOKEN_RE.findall(
        text
    )


def identifier(
    value: Any,
) -> str:
    if value in (
        None,
        "",
    ):
        return ""

    text = str(value).strip()

    if (
        text.endswith(".0")
        and text[:-2].isdigit()
    ):
        return text[:-2]

    return text


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def as_int(
    value: Any,
    default: int = 0,
) -> int:
    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        try:
            return int(
                float(value)
            )

        except (
            TypeError,
            ValueError,
        ):
            return default


def build_sources(
    company_rows: list[dict[str, str]],
    alias_rows: list[dict[str, str]],
) -> tuple[
    list[SourceAlias],
    dict[str, set[str]],
]:
    companies = {
        identifier(
            row.get("company_id")
        ): row
        for row in company_rows
        if identifier(
            row.get("company_id")
        )
    }

    sources: list[
        SourceAlias
    ] = []

    owners: dict[
        str,
        set[str],
    ] = {}

    seen: set[
        tuple[str, str]
    ] = set()

    def add(
        company_id: str,
        alias_text: str,
        alias_type: str,
    ) -> None:
        alias_text = clean(
            alias_text
        )

        company = companies.get(
            company_id
        )

        if (
            not company
            or not alias_text
        ):
            return

        normalized = normalize(
            alias_text
        )

        if (
            not normalized
            or sum(
                character.isalnum()
                for character in normalized
            )
            < 5
        ):
            return

        normalized_key = (
            normalized.casefold()
        )

        owners.setdefault(
            normalized_key,
            set(),
        ).add(
            company_id
        )

        identity_key = (
            company_id,
            normalized_key,
        )

        if identity_key in seen:
            return

        sources.append(
            SourceAlias(
                company_id=company_id,
                legal_name=clean(
                    company.get(
                        "legal_name",
                        "",
                    )
                ),
                alias_text=alias_text,
                alias_type=alias_type,
                normalized_alias=normalized,
            )
        )

        seen.add(
            identity_key
        )

    for row in alias_rows:
        add(
            identifier(
                row.get("company_id")
            ),
            row.get(
                "alias_text",
                "",
            ),
            row.get(
                "alias_type",
                "EXPLICIT_ALIAS",
            ),
        )

    for (
        company_id,
        company,
    ) in companies.items():
        add(
            company_id,
            company.get(
                "legal_name",
                "",
            ),
            "LEGAL_NAME",
        )

        add(
            company_id,
            company.get(
                "brand_name",
                "",
            ),
            "BRAND_NAME",
        )

    if not sources:
        raise ValueError(
            "Kullanılabilir alias bulunamadı."
        )

    return (
        sources,
        owners,
    )


def remove_legal(
    text: str,
    rng: random.Random,
) -> str:
    del rng

    retained_tokens = [
        token
        for token in tokens(text)
        if token.upper()
        not in LEGAL_TOKENS
    ]

    return (
        " ".join(retained_tokens)
        or text
    )


def drop_token(
    text: str,
    rng: random.Random,
) -> str:
    parts = tokens(
        text
    )

    if len(parts) < 2:
        return text

    choices = [
        index
        for index, token in enumerate(parts)
        if (
            len(token) >= 4
            and token.upper()
            not in LEGAL_TOKENS
        )
    ] or list(
        range(len(parts))
    )

    drop_index = rng.choice(
        choices
    )

    return " ".join(
        token
        for index, token in enumerate(parts)
        if index != drop_index
    )


def truncate(
    text: str,
    rng: random.Random,
) -> str:
    parts = tokens(
        text
    )

    mutable_indices = [
        index
        for index, token in enumerate(parts)
        if len(token) >= 5
    ]

    if not mutable_indices:
        return text

    changed = False

    for index in mutable_indices:
        if rng.random() < 0.65:
            token = parts[index]

            keep_length = rng.randint(
                3,
                min(
                    6,
                    len(token) - 1,
                ),
            )

            parts[index] = token[
                :keep_length
            ]

            changed = True

    if not changed:
        index = rng.choice(
            mutable_indices
        )

        parts[index] = parts[index][
            : min(
                5,
                len(parts[index]) - 1,
            )
        ]

    return " ".join(
        parts
    )


def delete_char(
    text: str,
    rng: random.Random,
) -> str:
    characters = list(
        text
    )

    choices = [
        index
        for index, character
        in enumerate(characters)
        if character.isalnum()
    ]

    if len(choices) <= 4:
        return text

    delete_count = (
        2
        if (
            len(choices) >= 12
            and rng.random() < 0.4
        )
        else 1
    )

    for index in sorted(
        rng.sample(
            choices,
            k=delete_count,
        ),
        reverse=True,
    ):
        del characters[index]

    return "".join(
        characters
    )


def transpose_char(
    text: str,
    rng: random.Random,
) -> str:
    characters = list(
        text
    )

    choices = [
        index
        for index in range(
            len(characters) - 1
        )
        if (
            characters[index].isalnum()
            and characters[
                index + 1
            ].isalnum()
            and characters[index]
            != characters[index + 1]
        )
    ]

    if not choices:
        return text

    index = rng.choice(
        choices
    )

    (
        characters[index],
        characters[index + 1],
    ) = (
        characters[index + 1],
        characters[index],
    )

    return "".join(
        characters
    )


def substitute_char(
    text: str,
    rng: random.Random,
) -> str:
    characters = list(
        text
    )

    choices = [
        index
        for index, character
        in enumerate(characters)
        if character.upper()
        in SUBSTITUTIONS
    ]

    if not choices:
        return delete_char(
            text,
            rng,
        )

    index = rng.choice(
        choices
    )

    original = characters[index]

    replacement = (
        SUBSTITUTIONS[
            original.upper()
        ]
    )

    characters[index] = (
        replacement.lower()
        if original.islower()
        else replacement
    )

    return "".join(
        characters
    )


def drop_vowels(
    text: str,
    rng: random.Random,
) -> str:
    parts = tokens(
        text
    )

    choices = [
        index
        for index, token in enumerate(parts)
        if len(token) >= 5
    ]

    if not choices:
        return text

    index = rng.choice(
        choices
    )

    token = parts[index]

    compact_token = (
        token[0]
        + "".join(
            character
            for character in token[1:]
            if character not in VOWELS
        )
    )

    if len(compact_token) >= 3:
        parts[index] = (
            compact_token
        )

    return " ".join(
        parts
    )


def merge_tokens(
    text: str,
    rng: random.Random,
) -> str:
    parts = tokens(
        text
    )

    if len(parts) < 2:
        return text.replace(
            " ",
            "",
        )

    index = rng.randrange(
        len(parts) - 1
    )

    merged_parts = (
        parts[:index]
        + [
            parts[index]
            + parts[index + 1]
        ]
        + parts[index + 2 :]
    )

    return " ".join(
        merged_parts
    )


def remove_spaces(
    text: str,
    rng: random.Random,
) -> str:
    del rng

    return "".join(
        tokens(text)
    )


def shuffle_tokens(
    text: str,
    rng: random.Random,
) -> str:
    parts = tokens(
        text
    )

    if len(parts) < 2:
        return text

    original_parts = list(
        parts
    )

    rng.shuffle(
        parts
    )

    if parts == original_parts:
        parts = (
            parts[1:]
            + parts[:1]
        )

    return " ".join(
        parts
    )


def initialism(
    text: str,
    rng: random.Random,
) -> str:
    parts = [
        token
        for token in tokens(text)
        if token.upper()
        not in LEGAL_TOKENS
    ]

    if len(parts) < 2:
        return text

    initials = "".join(
        token[0]
        for token in parts
    )

    if rng.random() < 0.5:
        return initials

    longest_token = max(
        parts,
        key=len,
    )

    return (
        f"{initials} "
        f"{longest_token[:5]}"
    )


def add_noise(
    text: str,
    rng: random.Random,
) -> str:
    noise = rng.choice(
        NOISE
    )

    if rng.random() < 0.3:
        reference = "".join(
            rng.choice(
                "0123456789"
            )
            for _ in range(
                rng.randint(
                    6,
                    10,
                )
            )
        )

        noise = (
            f"{noise} "
            f"REF {reference}"
        )

    if rng.random() < 0.2:
        return (
            f"{noise} {text}"
        )

    return (
        f"{text} {noise}"
    )


OPERATIONS: dict[
    str,
    Callable[
        [
            str,
            random.Random,
        ],
        str,
    ],
] = {
    "REMOVE_LEGAL": remove_legal,
    "DROP_TOKEN": drop_token,
    "TRUNCATE": truncate,
    "DELETE_CHAR": delete_char,
    "TRANSPOSE_CHAR": transpose_char,
    "SUBSTITUTE_CHAR": substitute_char,
    "DROP_VOWELS": drop_vowels,
    "MERGE_TOKENS": merge_tokens,
    "REMOVE_SPACES": remove_spaces,
    "SHUFFLE_TOKENS": shuffle_tokens,
    "INITIALISM": initialism,
    "ADD_NOISE": add_noise,
}


PIPELINES = (
    Pipeline(
        "LEGAL_DROP_TOKEN_DROP_TYPO",
        (
            "REMOVE_LEGAL",
            "DROP_TOKEN",
            "DELETE_CHAR",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "TOKEN_TRUNCATION_TYPO",
        (
            "REMOVE_LEGAL",
            "TRUNCATE",
            "SUBSTITUTE_CHAR",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "NO_SPACE_CHARACTER_ERRORS",
        (
            "REMOVE_LEGAL",
            "REMOVE_SPACES",
            "DELETE_CHAR",
            "TRANSPOSE_CHAR",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "TOKEN_SHUFFLE_TRUNCATION",
        (
            "REMOVE_LEGAL",
            "SHUFFLE_TOKENS",
            "TRUNCATE",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "VOWEL_DROP_TOKEN_DROP",
        (
            "REMOVE_LEGAL",
            "DROP_VOWELS",
            "DROP_TOKEN",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "MERGED_TOKEN_TYPO",
        (
            "REMOVE_LEGAL",
            "MERGE_TOKENS",
            "DELETE_CHAR",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "INITIALISM_WITH_HINT",
        (
            "REMOVE_LEGAL",
            "INITIALISM",
            "ADD_NOISE",
        ),
    ),
    Pipeline(
        "MULTI_TYPO",
        (
            "REMOVE_LEGAL",
            "DELETE_CHAR",
            "SUBSTITUTE_CHAR",
            "TRANSPOSE_CHAR",
            "ADD_NOISE",
        ),
    ),
)


def mutate(
    source: SourceAlias,
    pipeline: Pipeline,
    rng: random.Random,
) -> str:
    text = source.alias_text

    for operation_name in pipeline.operations:
        text = clean(
            OPERATIONS[
                operation_name
            ](
                text,
                rng,
            )
        )

        if not text:
            return ""

    return text


def check_backend(
    session: requests.Session,
    base_url: str,
    timeout: float,
) -> None:
    response = session.get(
        (
            f"{base_url.rstrip('/')}"
            "/openapi.json"
        ),
        timeout=timeout,
    )

    response.raise_for_status()

    paths = response.json().get(
        "paths",
        {},
    )

    required_path = (
        "/api/v1/matching/candidates"
    )

    if required_path not in paths:
        raise RuntimeError(
            "Matching endpointi bulunamadı. "
            f"Beklenen: {required_path}. "
            f"Mevcut yollar: {sorted(paths)}"
        )


def find_expected(
    candidates: list[dict[str, Any]],
    company_id: str,
) -> tuple[
    int,
    dict[str, Any] | None,
]:
    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if (
            identifier(
                candidate.get(
                    "company_id"
                )
            )
            == company_id
        ):
            return (
                rank,
                candidate,
            )

    return (
        0,
        None,
    )


def build_row(
    number: int,
    source: SourceAlias,
    pipeline: Pipeline,
    query_text: str,
    normalized_query: str,
    payload: dict[str, Any],
    expected_rank: int,
    expected: dict[str, Any],
    attempt: int,
    seed: int,
) -> dict[str, Any]:
    candidates = payload.get(
        "candidates",
        [],
    )

    predicted = (
        candidates[0]
        if candidates
        else {}
    )

    expected_score = as_float(
        expected.get(
            "final_score"
        )
    )

    predicted_score = as_float(
        predicted.get(
            "final_score"
        )
    )

    return {
        "benchmark_id": (
            f"HP-{number}"
        ),
        "benchmark_type": (
            "HARD_POSITIVE"
        ),
        "query_text": (
            query_text
        ),
        "normalized_query": (
            normalized_query
        ),
        "company_id": (
            source.company_id
        ),
        "expected_company_id": (
            source.company_id
        ),
        "expected_legal_name": (
            source.legal_name
        ),
        "expected_decision": (
            "MANUAL_REVIEW"
        ),
        "source_alias": (
            source.alias_text
        ),
        "source_alias_type": (
            source.alias_type
        ),
        "source_normalized_alias": (
            source.normalized_alias
        ),
        "perturbation_type": (
            pipeline.name
        ),
        "perturbation_operations": (
            "|".join(
                pipeline.operations
            )
        ),
        "expected_rank_before_catboost": (
            expected_rank
        ),
        "expected_rank_after_catboost": (
            as_int(
                expected.get(
                    "catboost_rank"
                )
            )
        ),
        "expected_retrieval_rank": (
            expected.get(
                "retrieval_rank",
                0,
            )
        ),
        "expected_retrieval_score": (
            expected.get(
                "retrieval_score",
                0.0,
            )
        ),
        "expected_reranker_score": (
            expected.get(
                "reranker_score",
                0.0,
            )
        ),
        "expected_adjusted_reranker_score": (
            expected.get(
                "adjusted_reranker_score",
                0.0,
            )
        ),
        "expected_final_score": (
            expected_score
        ),
        "expected_catboost_raw_score": (
            expected.get(
                "catboost_raw_score",
                0.0,
            )
        ),
        "predicted_company_id": (
            identifier(
                predicted.get(
                    "company_id"
                )
            )
        ),
        "predicted_legal_name": (
            predicted.get(
                "legal_name",
                "",
            )
        ),
        "predicted_alias": (
            predicted.get(
                "matched_alias",
                "",
            )
        ),
        "predicted_final_score": (
            predicted_score
        ),
        "score_gap_to_predicted": round(
            predicted_score
            - expected_score,
            6,
        ),
        "actual_decision": (
            payload.get(
                "decision",
                "",
            )
        ),
        "score_margin": (
            payload.get(
                "score_margin",
                0.0,
            )
        ),
        "ranking_mode": (
            payload.get(
                "ranking_mode",
                "",
            )
        ),
        "catboost_shadow_applied": (
            payload.get(
                "catboost_shadow_applied",
                False,
            )
        ),
        "catboost_shadow_changed_winner": (
            payload.get(
                "catboost_shadow_changed_winner",
                False,
            )
        ),
        "candidate_count": (
            len(candidates)
        ),
        "benchmark_valid": True,
        "strict_evaluation_eligible": True,
        "generation_attempt": (
            attempt
        ),
        "seed": seed,
        "top3_candidates_json": (
            json.dumps(
                [
                    {
                        "rank": rank,
                        "company_id": (
                            candidate.get(
                                "company_id"
                            )
                        ),
                        "legal_name": (
                            candidate.get(
                                "legal_name",
                                "",
                            )
                        ),
                        "final_score": (
                            candidate.get(
                                "final_score",
                                0.0,
                            )
                        ),
                        "catboost_rank": (
                            candidate.get(
                                "catboost_rank",
                                0,
                            )
                        ),
                    }
                    for rank, candidate
                    in enumerate(
                        candidates[:3],
                        start=1,
                    )
                ],
                ensure_ascii=False,
                separators=(
                    ",",
                    ":",
                ),
            )
        ),
    }


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        raise ValueError(
            "Hard-positive sorgu üretilemedi."
        )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fields: list[str] = []

    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(
                    field
                )

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def main() -> None:
    args = parse_args()

    validate_args(
        args
    )

    company_rows = read_csv(
        args.companies_path,
        "Şirket CSV",
    )

    alias_rows = read_csv(
        args.aliases_path,
        "Alias CSV",
    )

    (
        sources,
        owners,
    ) = build_sources(
        company_rows,
        alias_rows,
    )

    rng = random.Random(
        args.seed
    )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "hard-positive-generator/1.0"
            ),
        }
    )

    accepted: list[
        dict[str, Any]
    ] = []

    seen_queries: set[
        str
    ] = set()

    company_counts: Counter[
        str
    ] = Counter()

    rejections: Counter[
        str
    ] = Counter()

    accepted_pipelines: Counter[
        str
    ] = Counter()

    try:
        check_backend(
            session,
            args.backend_url,
            args.timeout,
        )

        print(
            "Backend ve matching endpointi hazır."
        )

        print(
            "Kaynak alias sayısı: "
            f"{len(sources)}"
        )

        print(
            "Hedef hard-positive sayısı: "
            f"{args.target_count}"
        )

        for attempt in range(
            1,
            args.max_attempts + 1,
        ):
            if (
                len(accepted)
                >= args.target_count
            ):
                break

            source = rng.choice(
                sources
            )

            if (
                company_counts[
                    source.company_id
                ]
                >= args.max_per_company
            ):
                rejections[
                    "COMPANY_LIMIT"
                ] += 1

                continue

            pipeline = rng.choice(
                PIPELINES
            )

            query_text = mutate(
                source,
                pipeline,
                rng,
            )

            normalized_query = (
                normalize(query_text)
                if query_text
                else ""
            )

            if not normalized_query:
                rejections[
                    "EMPTY"
                ] += 1

                continue

            if (
                normalized_query.casefold()
                == source.normalized_alias.casefold()
            ):
                rejections[
                    "UNCHANGED"
                ] += 1

                continue

            query_key = (
                normalized_query.casefold()
            )

            if query_key in seen_queries:
                rejections[
                    "DUPLICATE"
                ] += 1

                continue

            collision_owners = owners.get(
                query_key,
                set(),
            )

            if (
                collision_owners
                and collision_owners
                != {
                    source.company_id
                }
            ):
                rejections[
                    "ALIAS_COLLISION"
                ] += 1

                continue

            try:
                response = session.post(
                    (
                        f"{args.backend_url.rstrip('/')}"
                        "/api/v1/matching/candidates"
                    ),
                    json={
                        "text": query_text,
                        "limit": (
                            args.candidate_limit
                        ),
                    },
                    timeout=args.timeout,
                )

                response.raise_for_status()

                payload = response.json()

            except Exception:
                rejections[
                    "API_ERROR"
                ] += 1

                continue

            candidates = payload.get(
                "candidates",
                [],
            )

            if not isinstance(
                candidates,
                list,
            ):
                rejections[
                    "INVALID_CANDIDATES"
                ] += 1

                continue

            (
                expected_rank,
                expected,
            ) = find_expected(
                candidates,
                source.company_id,
            )

            if expected_rank == 0:
                rejections[
                    "EXPECTED_MISSING"
                ] += 1

                continue

            if (
                expected_rank
                < args.minimum_expected_rank
            ):
                rejections[
                    "TOO_EASY"
                ] += 1

                continue

            if (
                expected_rank
                > args.maximum_expected_rank
            ):
                rejections[
                    "RANK_TOO_LOW"
                ] += 1

                continue

            if expected is None:
                rejections[
                    "EXPECTED_NONE"
                ] += 1

                continue

            seen_queries.add(
                query_key
            )

            company_counts[
                source.company_id
            ] += 1

            accepted_pipelines[
                pipeline.name
            ] += 1

            accepted.append(
                build_row(
                    len(accepted) + 1,
                    source,
                    pipeline,
                    query_text,
                    normalized_query,
                    payload,
                    expected_rank,
                    expected,
                    attempt,
                    args.seed,
                )
            )

            if (
                len(accepted) == 1
                or len(accepted) % 10 == 0
            ):
                print(
                    f"Kabul: {len(accepted)}/"
                    f"{args.target_count} | "
                    f"Deneme: {attempt} | "
                    f"Son sıra: {expected_rank}"
                )

            if attempt % 500 == 0:
                print(
                    f"Deneme {attempt} | "
                    f"Kabul {len(accepted)} | "
                    "Too easy "
                    f"{rejections['TOO_EASY']} | "
                    "Missing "
                    f"{rejections['EXPECTED_MISSING']}"
                )

    finally:
        session.close()

    write_csv(
        args.output,
        accepted,
    )

    rank_counts = Counter(
        int(
            row[
                "expected_rank_before_catboost"
            ]
        )
        for row in accepted
    )

    changed_count = sum(
        bool(
            row[
                "catboost_shadow_changed_winner"
            ]
        )
        for row in accepted
    )

    print()
    print(
        "Hard-positive benchmark oluşturuldu."
    )

    print(
        f"Üretilen sorgu: {len(accepted)}"
    )

    print(
        f"Farklı şirket: {len(company_counts)}"
    )

    print(
        "Beklenen sıra dağılımı: "
        f"{dict(sorted(rank_counts.items()))}"
    )

    print(
        "CatBoost winner değiştirdi: "
        f"{changed_count}"
    )

    print(
        "Pipeline dağılımı: "
        f"{dict(accepted_pipelines)}"
    )

    print(
        "Ret nedenleri: "
        f"{dict(rejections)}"
    )

    print(
        f"Dosya: {args.output}"
    )

    if (
        len(accepted)
        < args.target_count
    ):
        print(
            "Uyarı: Hedef sayıya ulaşılamadı. "
            "--max-attempts veya --max-per-company "
            "değerini artırın."
        )


if __name__ == "__main__":
    main()