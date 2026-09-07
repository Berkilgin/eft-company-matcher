from __future__ import annotations

import argparse
import csv
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rapidfuzz import fuzz, process

from app.normalization.normalizer import EFTNormalizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COMPANIES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "companies.csv"
)

DEFAULT_ALIASES_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aliases.csv"
)

DEFAULT_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "negative_benchmark_queries.csv"
)


WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


CATEGORY_WEIGHTS = {
    "SYNTHETIC_COMPANY": 0.40,
    "PERSON_PAYMENT": 0.25,
    "GENERIC_PAYMENT": 0.20,
    "REFERENCE_ONLY": 0.15,
}


COMPANY_PREFIXES = (
    "ARVENA",
    "BELVORA",
    "CERANTA",
    "DORVEX",
    "ELMORA",
    "FENRIX",
    "GAVENTA",
    "HELVARA",
    "ILVORA",
    "JUNTERA",
    "KERVONA",
    "LUMERA",
    "MERVEX",
    "NORDIVA",
    "ORVENA",
    "PENTORA",
    "QUANTERA",
    "RAVENTA",
    "SELVORA",
    "TRENOVA",
    "ULMERA",
    "VENTARA",
    "WELVORA",
    "XENTORA",
    "YALVENA",
    "ZORVEX",
)


COMPANY_ACTIVITY_TOKENS = (
    "TEKNOLOJI",
    "YAZILIM",
    "LOJISTIK",
    "ENERJI",
    "OTOMASYON",
    "ROBOTIK",
    "DANISMANLIK",
    "MIMARLIK",
    "GIDA",
    "MOBILYA",
    "MEDYA",
    "ELEKTRONIK",
    "MAKINE",
    "SAVUNMA",
    "HAVACILIK",
    "TASARIM",
    "URETIM",
    "MUHENDISLIK",
    "INSAAT",
    "TICARET",
    "DIGITAL",
    "SISTEMLERI",
)


COMPANY_SUFFIXES = (
    "LIMITED",
    "LIMITED SIRKETI",
    "ANONIM SIRKETI",
    "LLC",
    "LTD",
    "GMBH",
    "SAS",
    "BV",
    "INC",
    "CORPORATION",
)


PAYMENT_NOISE = (
    "FATURA ODEMESI",
    "ODEME",
    "HAVALE",
    "EFT",
    "HIZMET BEDELI",
    "AVANS ODEMESI",
    "SIPARIS BEDELI",
    "MAL BEDELI",
    "REF",
)


FIRST_NAMES = (
    "AHMET",
    "MEHMET",
    "MUSTAFA",
    "ALI",
    "MURAT",
    "EMRE",
    "CAN",
    "BURAK",
    "KEREM",
    "ONUR",
    "AYSE",
    "FATMA",
    "ZEYNEP",
    "ELIF",
    "MERVE",
    "ESRA",
    "SELIN",
    "DERYA",
    "DENIZ",
    "ECE",
)


LAST_NAMES = (
    "YILMAZ",
    "KAYA",
    "DEMIR",
    "CELIK",
    "SAHIN",
    "YILDIZ",
    "AYDIN",
    "OZTURK",
    "ARSLAN",
    "DOGAN",
    "KILIC",
    "ASLAN",
    "CETIN",
    "KOC",
    "KURT",
    "OZDEMIR",
    "AKSOY",
    "ERDEM",
    "POLAT",
    "TEKIN",
)


PERSON_PAYMENT_REASONS = (
    "KIRA",
    "BORC ODEMESI",
    "MASRAF IADE",
    "AVANS",
    "ORTAK GIDER",
    "YEMEK BEDELI",
    "SEYAHAT MASRAFI",
    "DEPOZITO",
    "EMANET",
    "HARCAMA IADE",
)


GENERIC_PAYMENT_TEXTS = (
    "TEMMUZ AYI KIRA ODEMESI",
    "AGUSTOS AYI AIDAT ODEMESI",
    "ELEKTRIK FATURASI ODEMESI",
    "SU FATURASI ODEMESI",
    "INTERNET FATURASI",
    "DOGALGAZ FATURASI",
    "PERSONEL YEMEK BEDELI",
    "OFIS MASRAF ODEMESI",
    "GENEL GIDER ODEMESI",
    "DEPOZITO IADE ODEMESI",
    "KARGO MASRAFI",
    "SEYAHAT MASRAFI",
    "ARAC YAKIT GIDERI",
    "BAKIM ONARIM BEDELI",
    "VERGI HARCI ODEMESI",
    "NOTER MASRAFI",
    "SIGORTA PRIM ODEMESI",
    "ABONELIK BEDELI",
    "EGITIM KATILIM BEDELI",
    "TOPLANTI ORGANIZASYON GIDERI",
)


normalizer = EFTNormalizer()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "UNKNOWN kararını ölçmek için negatif ve "
            "veri tabanı dışı benchmark sorguları üretir."
        )
    )

    parser.add_argument(
        "--companies-path",
        type=Path,
        default=DEFAULT_COMPANIES_PATH,
    )

    parser.add_argument(
        "--aliases-path",
        type=Path,
        default=DEFAULT_ALIASES_PATH,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )

    parser.add_argument(
        "--total",
        type=int,
        default=400,
        help=(
            "Üretilecek toplam negatif sorgu sayısı."
        ),
    )

    parser.add_argument(
        "--max-known-similarity",
        type=float,
        default=0.72,
        help=(
            "Negatif sorgunun bilinen aliaslarla izin verilen "
            "maksimum WRatio benzerliği."
        ),
    )

    parser.add_argument(
        "--max-generation-attempts",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.total < 1:
        raise ValueError(
            "--total en az 1 olmalıdır."
        )

    if not (
        0.0
        <= arguments.max_known_similarity
        <= 1.0
    ):
        raise ValueError(
            "--max-known-similarity 0 ile 1 "
            "arasında olmalıdır."
        )

    if arguments.max_generation_attempts < 1:
        raise ValueError(
            "--max-generation-attempts en az "
            "1 olmalıdır."
        )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"CSV dosyası bulunamadı: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def clean_spaces(
    text: str,
) -> str:
    return WHITESPACE_PATTERN.sub(
        " ",
        text,
    ).strip()


def normalize_text(
    text: str,
) -> str:
    normalization = normalizer.normalize(
        text
    )

    return (
        normalization["core_text"]
        or normalization["ascii_text"]
    ).strip()


def build_known_alias_index(
    companies: list[dict[str, str]],
    aliases: list[dict[str, str]],
) -> tuple[
    list[str],
    dict[str, set[str]],
    dict[str, set[str]],
]:
    """
    Bilinen normalize aliasları ve sahiplerini oluşturur.

    Dönen değerler:
    - Benzersiz normalize alias listesi
    - Normalize alias -> company_id kümesi
    - Normalize alias -> orijinal alias kümesi
    """

    alias_company_ids: dict[
        str,
        set[str],
    ] = {}

    alias_original_texts: dict[
        str,
        set[str],
    ] = {}

    def add_alias(
        company_id: str,
        alias_text: str,
    ) -> None:
        normalized_alias = normalize_text(
            alias_text
        )

        if not normalized_alias:
            return

        alias_company_ids.setdefault(
            normalized_alias,
            set(),
        ).add(
            company_id
        )

        alias_original_texts.setdefault(
            normalized_alias,
            set(),
        ).add(
            alias_text.strip()
        )

    for alias_row in aliases:
        company_id = alias_row.get(
            "company_id",
            "",
        ).strip()

        alias_text = alias_row.get(
            "alias_text",
            "",
        ).strip()

        if company_id and alias_text:
            add_alias(
                company_id,
                alias_text,
            )

    for company_row in companies:
        company_id = company_row.get(
            "company_id",
            "",
        ).strip()

        if not company_id:
            continue

        for column_name in (
            "legal_name",
            "brand_name",
        ):
            company_name = company_row.get(
                column_name,
                "",
            ).strip()

            if company_name:
                add_alias(
                    company_id,
                    company_name,
                )

    known_aliases = sorted(
        alias_company_ids
    )

    if not known_aliases:
        raise ValueError(
            "Bilinen şirket alias indeksi boş."
        )

    return (
        known_aliases,
        alias_company_ids,
        alias_original_texts,
    )


def calculate_category_quotas(
    total_count: int,
) -> dict[str, int]:
    quotas: dict[str, int] = {}

    allocated_count = 0

    category_names = list(
        CATEGORY_WEIGHTS
    )

    for category_name in category_names[
        :-1
    ]:
        category_count = int(
            total_count
            * CATEGORY_WEIGHTS[
                category_name
            ]
        )

        quotas[
            category_name
        ] = category_count

        allocated_count += (
            category_count
        )

    last_category = category_names[-1]

    quotas[
        last_category
    ] = (
        total_count
        - allocated_count
    )

    return quotas


def random_reference_number(
    rng: random.Random,
    minimum_digits: int = 6,
    maximum_digits: int = 12,
) -> str:
    digit_count = rng.randint(
        minimum_digits,
        maximum_digits,
    )

    return "".join(
        rng.choice(
            "0123456789"
        )
        for _ in range(
            digit_count
        )
    )


def generate_synthetic_company_query(
    rng: random.Random,
) -> str:
    prefix = rng.choice(
        COMPANY_PREFIXES
    )

    activity_token_count = rng.choice(
        (
            1,
            1,
            2,
        )
    )

    activity_tokens = rng.sample(
        COMPANY_ACTIVITY_TOKENS,
        k=activity_token_count,
    )

    company_suffix = rng.choice(
        COMPANY_SUFFIXES
    )

    company_name = " ".join(
        (
            prefix,
            *activity_tokens,
            company_suffix,
        )
    )

    payment_noise = rng.choice(
        PAYMENT_NOISE
    )

    reference_number = (
        random_reference_number(
            rng
        )
    )

    query_variants = (
        (
            f"{company_name} "
            f"{payment_noise} "
            f"{reference_number}"
        ),
        (
            f"{prefix} "
            f"{' '.join(activity_tokens)} "
            f"{payment_noise}"
        ),
        (
            f"{company_name} "
            f"REF {reference_number}"
        ),
    )

    return clean_spaces(
        rng.choice(
            query_variants
        )
    )


def generate_person_payment_query(
    rng: random.Random,
) -> str:
    first_name = rng.choice(
        FIRST_NAMES
    )

    last_name = rng.choice(
        LAST_NAMES
    )

    payment_reason = rng.choice(
        PERSON_PAYMENT_REASONS
    )

    reference_number = (
        random_reference_number(
            rng
        )
    )

    query_variants = (
        (
            f"{first_name} {last_name} "
            f"{payment_reason}"
        ),
        (
            f"{first_name} {last_name} "
            f"{payment_reason} "
            f"REF {reference_number}"
        ),
        (
            f"{payment_reason} "
            f"{first_name} {last_name}"
        ),
    )

    return clean_spaces(
        rng.choice(
            query_variants
        )
    )


def generate_generic_payment_query(
    rng: random.Random,
) -> str:
    generic_text = rng.choice(
        GENERIC_PAYMENT_TEXTS
    )

    reference_number = (
        random_reference_number(
            rng
        )
    )

    query_variants = (
        generic_text,
        (
            f"{generic_text} "
            f"REF {reference_number}"
        ),
        (
            f"{reference_number} "
            f"{generic_text}"
        ),
    )

    return clean_spaces(
        rng.choice(
            query_variants
        )
    )


def generate_reference_only_query(
    rng: random.Random,
) -> str:
    reference_number = (
        random_reference_number(
            rng,
            minimum_digits=8,
            maximum_digits=16,
        )
    )

    secondary_reference = (
        random_reference_number(
            rng,
            minimum_digits=6,
            maximum_digits=10,
        )
    )

    random_code = "".join(
        rng.choice(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        for _ in range(
            rng.randint(
                5,
                10,
            )
        )
    )

    query_variants = (
        (
            f"EFT REF {reference_number}"
        ),
        (
            f"HAVALE REF {reference_number} "
            f"ISLEM {secondary_reference}"
        ),
        (
            f"ODEME {random_code} "
            f"{reference_number}"
        ),
        (
            f"REF {reference_number} "
            f"DEKONT {secondary_reference}"
        ),
        (
            f"ISLEM NO {reference_number}"
        ),
    )

    return clean_spaces(
        rng.choice(
            query_variants
        )
    )


def generate_query_by_category(
    category: str,
    rng: random.Random,
) -> str:
    if category == "SYNTHETIC_COMPANY":
        return generate_synthetic_company_query(
            rng
        )

    if category == "PERSON_PAYMENT":
        return generate_person_payment_query(
            rng
        )

    if category == "GENERIC_PAYMENT":
        return generate_generic_payment_query(
            rng
        )

    if category == "REFERENCE_ONLY":
        return generate_reference_only_query(
            rng
        )

    raise ValueError(
        f"Bilinmeyen negatif kategori: {category}"
    )


def find_closest_known_alias(
    normalized_query: str,
    known_aliases: list[str],
) -> tuple[
    str,
    float,
]:
    if not normalized_query:
        return (
            "",
            0.0,
        )

    closest_match = process.extractOne(
        normalized_query,
        known_aliases,
        scorer=fuzz.WRatio,
        processor=None,
        score_cutoff=0,
    )

    if closest_match is None:
        return (
            "",
            0.0,
        )

    closest_alias = str(
        closest_match[0]
    )

    similarity = (
        float(
            closest_match[1]
        )
        / 100.0
    )

    return (
        closest_alias,
        max(
            0.0,
            min(
                1.0,
                similarity,
            ),
        ),
    )


def build_negative_row(
    *,
    benchmark_id: int,
    category: str,
    query_text: str,
    normalized_query: str,
    closest_alias: str,
    closest_similarity: float,
    generation_attempts: int,
    known_alias_company_ids: dict[
        str,
        set[str],
    ],
    known_alias_original_texts: dict[
        str,
        set[str],
    ],
    seed: int,
) -> dict[str, Any]:
    closest_company_ids = sorted(
        known_alias_company_ids.get(
            closest_alias,
            set(),
        )
    )

    closest_original_aliases = sorted(
        known_alias_original_texts.get(
            closest_alias,
            set(),
        )
    )

    return {
        "benchmark_id": benchmark_id,
        "benchmark_type": "NEGATIVE",
        "negative_category": category,
        "query_text": query_text,
        "normalized_query": normalized_query,
        "expected_decision": "UNKNOWN",
        "expected_company_id": "",
        "is_out_of_database": (
            category
            == "SYNTHETIC_COMPANY"
        ),
        "is_person_payment": (
            category
            == "PERSON_PAYMENT"
        ),
        "is_reference_only": (
            category
            == "REFERENCE_ONLY"
        ),
        "closest_known_alias": (
            closest_alias
        ),
        "closest_known_original_aliases": (
            "|".join(
                closest_original_aliases[
                    :10
                ]
            )
        ),
        "closest_known_company_ids": (
            "|".join(
                closest_company_ids
            )
        ),
        "closest_known_similarity": round(
            closest_similarity,
            6,
        ),
        "benchmark_valid": True,
        "strict_evaluation_eligible": True,
        "generation_attempts": (
            generation_attempts
        ),
        "seed": seed,
    }


def generate_negative_rows(
    *,
    total_count: int,
    rng: random.Random,
    known_aliases: list[str],
    known_alias_company_ids: dict[
        str,
        set[str],
    ],
    known_alias_original_texts: dict[
        str,
        set[str],
    ],
    maximum_similarity: float,
    maximum_attempts: int,
    seed: int,
) -> list[dict[str, Any]]:
    category_quotas = (
        calculate_category_quotas(
            total_count
        )
    )

    generated_rows: list[
        dict[str, Any]
    ] = []

    generated_query_keys: set[
        str
    ] = set()

    benchmark_id = 1

    for (
        category,
        category_count,
    ) in category_quotas.items():
        for _ in range(
            category_count
        ):
            selected_result: (
                tuple[
                    str,
                    str,
                    str,
                    float,
                    int,
                ]
                | None
            ) = None

            for attempt_number in range(
                1,
                maximum_attempts + 1,
            ):
                query_text = (
                    generate_query_by_category(
                        category,
                        rng,
                    )
                )

                normalized_query = normalize_text(
                    query_text
                )

                duplicate_key = (
                    query_text.casefold()
                )

                if (
                    duplicate_key
                    in generated_query_keys
                ):
                    continue

                (
                    closest_alias,
                    closest_similarity,
                ) = find_closest_known_alias(
                    normalized_query,
                    known_aliases,
                )

                # Referans-only sorguların normalize metni
                # boş olabilir. Bu beklenen bir UNKNOWN
                # örneğidir ve geçerli kabul edilir.
                reference_only_empty_query = (
                    category
                    == "REFERENCE_ONLY"
                    and not normalized_query
                )

                similarity_is_safe = (
                    closest_similarity
                    <= maximum_similarity
                )

                exact_known_collision = (
                    bool(normalized_query)
                    and normalized_query
                    in known_alias_company_ids
                )

                if (
                    (
                        similarity_is_safe
                        or reference_only_empty_query
                    )
                    and not exact_known_collision
                ):
                    selected_result = (
                        query_text,
                        normalized_query,
                        closest_alias,
                        closest_similarity,
                        attempt_number,
                    )

                    break

            if selected_result is None:
                raise RuntimeError(
                    "Güvenli negatif sorgu üretilemedi. "
                    f"Kategori: {category}. "
                    "--max-known-similarity değerini "
                    "bir miktar yükseltin veya "
                    "--max-generation-attempts değerini artırın."
                )

            (
                query_text,
                normalized_query,
                closest_alias,
                closest_similarity,
                generation_attempts,
            ) = selected_result

            generated_query_keys.add(
                query_text.casefold()
            )

            generated_rows.append(
                build_negative_row(
                    benchmark_id=(
                        benchmark_id
                    ),
                    category=category,
                    query_text=query_text,
                    normalized_query=(
                        normalized_query
                    ),
                    closest_alias=(
                        closest_alias
                    ),
                    closest_similarity=(
                        closest_similarity
                    ),
                    generation_attempts=(
                        generation_attempts
                    ),
                    known_alias_company_ids=(
                        known_alias_company_ids
                    ),
                    known_alias_original_texts=(
                        known_alias_original_texts
                    ),
                    seed=seed,
                )
            )

            benchmark_id += 1

    rng.shuffle(
        generated_rows
    )

    # Karıştırma sonrasında benchmark_id değerlerini
    # sıralı hâle getir.
    for new_id, row in enumerate(
        generated_rows,
        start=1,
    ):
        row["benchmark_id"] = (
            new_id
        )

    return generated_rows


def write_output(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "benchmark_id",
        "benchmark_type",
        "negative_category",
        "query_text",
        "normalized_query",
        "expected_decision",
        "expected_company_id",
        "is_out_of_database",
        "is_person_payment",
        "is_reference_only",
        "closest_known_alias",
        "closest_known_original_aliases",
        "closest_known_company_ids",
        "closest_known_similarity",
        "benchmark_valid",
        "strict_evaluation_eligible",
        "generation_attempts",
        "seed",
    ]

    with path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def print_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    category_counts = Counter(
        row[
            "negative_category"
        ]
        for row in rows
    )

    similarities = [
        float(
            row[
                "closest_known_similarity"
            ]
        )
        for row in rows
    ]

    regenerated_count = sum(
        int(
            row[
                "generation_attempts"
            ]
        )
        > 1
        for row in rows
    )

    empty_normalized_count = sum(
        not str(
            row[
                "normalized_query"
            ]
        ).strip()
        for row in rows
    )

    maximum_similarity = (
        max(similarities)
        if similarities
        else 0.0
    )

    average_similarity = (
        sum(similarities)
        / len(similarities)
        if similarities
        else 0.0
    )

    print()
    print(
        "Negatif benchmark veri seti oluşturuldu."
    )

    print(
        f"Toplam sorgu: {len(rows)}"
    )

    for category_name in CATEGORY_WEIGHTS:
        print(
            f"{category_name}: "
            f"{category_counts[category_name]}"
        )

    print(
        "Birden fazla denemede üretilen: "
        f"{regenerated_count}"
    )

    print(
        "Normalize metni boş sorgu: "
        f"{empty_normalized_count}"
    )

    print(
        "Ortalama en yakın alias benzerliği: "
        f"{average_similarity:.4f}"
    )

    print(
        "Maksimum en yakın alias benzerliği: "
        f"{maximum_similarity:.4f}"
    )

    print(
        f"Dosya: {output_path}"
    )


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    companies = read_csv(
        arguments.companies_path
    )

    aliases = read_csv(
        arguments.aliases_path
    )

    (
        known_aliases,
        known_alias_company_ids,
        known_alias_original_texts,
    ) = build_known_alias_index(
        companies=companies,
        aliases=aliases,
    )

    rng = random.Random(
        arguments.seed
    )

    rows = generate_negative_rows(
        total_count=arguments.total,
        rng=rng,
        known_aliases=known_aliases,
        known_alias_company_ids=(
            known_alias_company_ids
        ),
        known_alias_original_texts=(
            known_alias_original_texts
        ),
        maximum_similarity=(
            arguments.max_known_similarity
        ),
        maximum_attempts=(
            arguments.max_generation_attempts
        ),
        seed=arguments.seed,
    )

    write_output(
        path=arguments.output,
        rows=rows,
    )

    print_summary(
        rows=rows,
        output_path=arguments.output,
    )


if __name__ == "__main__":
    main()