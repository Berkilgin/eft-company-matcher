from __future__ import annotations

import argparse
import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

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
    / "benchmark_queries.csv"
)


NOISE_SUFFIXES = (
    "FATURA ODEMESI",
    "EFT ODEMESI",
    "HAVALE BEDELI",
    "ODEME",
    "FATURA",
    "ODM",
    "EFT REF 381729",
    "HAVALE REF 729184",
)


VOWELS = frozenset(
    "aeıioöuüAEIİOÖUÜ"
)


WHITESPACE_PATTERN = re.compile(
    r"\s+"
)


LATIN_CHARACTER_PATTERN = re.compile(
    r"[A-Za-zÇĞİÖŞÜçğıöşü]"
)


ALIAS_TYPE_PRIORITY = {
    "BRAND_NAME": 0,
    "GENERATED_SHORT_NAME": 1,
    "GLEIF_OTHER_NAME": 2,
    "GLEIF_TRANSLITERATED_NAME": 3,
    "LEGAL_NAME": 4,
    "GLEIF_LEGAL_NAME": 5,
}


COLLISION_REASON = (
    "NORMALIZED_QUERY_MATCHES_OTHER_COMPANY_ALIAS"
)


normalizer = EFTNormalizer()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Şirket aliaslarından bozulmuş EFT açıklamaları "
            "üretir ve alias çakışmalarını kontrol eder."
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
        "--max-companies",
        type=int,
        default=700,
    )

    parser.add_argument(
        "--variants-per-company",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--max-regeneration-attempts",
        type=int,
        default=8,
        help=(
            "Başka şirket aliasıyla çakışan sorgunun "
            "yeniden üretileceği maksimum deneme sayısı."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


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
    result = normalizer.normalize(text)

    return (
        result["core_text"]
        or result["ascii_text"]
    ).strip()


def append_noise(
    text: str,
    rng: random.Random,
) -> str:
    return clean_spaces(
        f"{text} {rng.choice(NOISE_SUFFIXES)}"
    )


def delete_character(
    text: str,
    rng: random.Random,
) -> str:
    candidate_indices = [
        index
        for index, character in enumerate(text)
        if character.isalnum()
    ]

    if len(candidate_indices) <= 4:
        return text

    selected_index = rng.choice(
        candidate_indices
    )

    return clean_spaces(
        text[:selected_index]
        + text[selected_index + 1:]
    )


def swap_adjacent_characters(
    text: str,
    rng: random.Random,
) -> str:
    candidate_indices = [
        index
        for index in range(len(text) - 1)
        if (
            text[index].isalnum()
            and text[index + 1].isalnum()
        )
    ]

    if not candidate_indices:
        return text

    selected_index = rng.choice(
        candidate_indices
    )

    characters = list(text)

    (
        characters[selected_index],
        characters[selected_index + 1],
    ) = (
        characters[selected_index + 1],
        characters[selected_index],
    )

    return clean_spaces(
        "".join(characters)
    )


def remove_vowels(
    text: str,
    rng: random.Random,
) -> str:
    tokens = text.split()

    eligible_token_indices = [
        index
        for index, token in enumerate(tokens)
        if (
            len(token) >= 5
            and any(
                character in VOWELS
                for character in token[1:]
            )
        )
    ]

    if not eligible_token_indices:
        return delete_character(
            text,
            rng,
        )

    token_index = rng.choice(
        eligible_token_indices
    )

    token = tokens[token_index]

    vowel_indices = [
        index
        for index, character in enumerate(token)
        if (
            index > 0
            and character in VOWELS
        )
    ]

    if not vowel_indices:
        return text

    vowel_index = rng.choice(
        vowel_indices
    )

    tokens[token_index] = (
        token[:vowel_index]
        + token[vowel_index + 1:]
    )

    return clean_spaces(
        " ".join(tokens)
    )


def abbreviate_tokens(
    text: str,
) -> str:
    tokens = text.split()

    if not tokens:
        return text

    abbreviated_tokens: list[str] = []

    for token in tokens:
        if len(token) <= 4:
            abbreviated_tokens.append(
                token
            )
            continue

        target_length = max(
            3,
            int(
                round(
                    len(token) * 0.65
                )
            ),
        )

        target_length = min(
            target_length,
            len(token),
        )

        abbreviated_tokens.append(
            token[:target_length]
        )

    return clean_spaces(
        " ".join(abbreviated_tokens)
    )


def remove_spaces(
    text: str,
) -> str:
    tokens = text.split()

    if len(tokens) < 2:
        return text

    return "".join(tokens)


def is_latin_dominant(
    text: str,
) -> bool:
    alphanumeric_characters = [
        character
        for character in text
        if character.isalnum()
    ]

    if not alphanumeric_characters:
        return False

    latin_character_count = len(
        LATIN_CHARACTER_PATTERN.findall(
            text
        )
    )

    return (
        latin_character_count
        / len(alphanumeric_characters)
        >= 0.60
    )


def is_ambiguous_query(
    normalized_query: str,
) -> bool:
    if not normalized_query:
        return True

    compact_query = normalized_query.replace(
        " ",
        "",
    )

    if len(compact_query) < 5:
        return True

    tokens = normalized_query.split()

    if (
        len(tokens) == 1
        and len(tokens[0]) <= 4
    ):
        return True

    return False


def make_variant(
    alias: str,
    difficulty: str,
    rng: random.Random,
) -> tuple[str, str]:
    if difficulty == "EASY":
        return (
            append_noise(
                alias,
                rng,
            ),
            "banking_noise",
        )

    if difficulty == "MEDIUM":
        operation = rng.choice(
            (
                delete_character,
                swap_adjacent_characters,
                remove_vowels,
            )
        )

        corrupted_alias = operation(
            alias,
            rng,
        )

        return (
            append_noise(
                corrupted_alias,
                rng,
            ),
            operation.__name__,
        )

    if not is_latin_dominant(alias):
        operation = rng.choice(
            (
                delete_character,
                swap_adjacent_characters,
            )
        )

        corrupted_alias = operation(
            alias,
            rng,
        )

        return (
            append_noise(
                corrupted_alias,
                rng,
            ),
            (
                "non_latin_"
                f"{operation.__name__}"
            ),
        )

    if len(alias.split()) >= 2:
        primary_operation_name, primary_operation = (
            rng.choice(
                (
                    (
                        "abbreviate_tokens",
                        abbreviate_tokens,
                    ),
                    (
                        "remove_spaces",
                        remove_spaces,
                    ),
                )
            )
        )

        corrupted_alias = primary_operation(
            alias
        )

    else:
        primary_operation_name = (
            "remove_vowels"
        )

        corrupted_alias = remove_vowels(
            alias,
            rng,
        )

    secondary_operation = rng.choice(
        (
            delete_character,
            swap_adjacent_characters,
            remove_vowels,
        )
    )

    secondary_corrupted_alias = (
        secondary_operation(
            corrupted_alias,
            rng,
        )
    )

    compact_secondary_alias = "".join(
        character
        for character in secondary_corrupted_alias
        if character.isalnum()
    )

    if len(compact_secondary_alias) >= 5:
        corrupted_alias = (
            secondary_corrupted_alias
        )

        corruption_type = (
            f"{primary_operation_name}"
            f"+{secondary_operation.__name__}"
        )

    else:
        corruption_type = (
            primary_operation_name
        )

    return (
        append_noise(
            corrupted_alias,
            rng,
        ),
        corruption_type,
    )


def choose_aliases(
    aliases: list[dict[str, str]],
) -> list[dict[str, str]]:
    unique_aliases: dict[
        str,
        dict[str, str],
    ] = {}

    for alias_row in aliases:
        alias_text = alias_row.get(
            "alias_text",
            "",
        ).strip()

        if len(alias_text) < 3:
            continue

        normalized_key = normalize_text(
            alias_text
        )

        if not normalized_key:
            continue

        current_alias = unique_aliases.get(
            normalized_key
        )

        if current_alias is None:
            unique_aliases[
                normalized_key
            ] = alias_row
            continue

        new_priority = (
            ALIAS_TYPE_PRIORITY.get(
                alias_row.get(
                    "alias_type",
                    "",
                ),
                100,
            )
        )

        current_priority = (
            ALIAS_TYPE_PRIORITY.get(
                current_alias.get(
                    "alias_type",
                    "",
                ),
                100,
            )
        )

        if new_priority < current_priority:
            unique_aliases[
                normalized_key
            ] = alias_row

    selected_aliases = list(
        unique_aliases.values()
    )

    selected_aliases.sort(
        key=lambda row: (
            ALIAS_TYPE_PRIORITY.get(
                row.get(
                    "alias_type",
                    "",
                ),
                100,
            ),
            abs(
                len(
                    row.get(
                        "alias_text",
                        "",
                    )
                )
                - 20
            ),
        )
    )

    return selected_aliases


def add_alias_owner(
    ownership_index: dict[
        str,
        dict[str, set[str]],
    ],
    *,
    company_id: str,
    alias_text: str,
) -> None:
    normalized_alias = normalize_text(
        alias_text
    )

    if not normalized_alias:
        return

    company_owners = ownership_index.setdefault(
        normalized_alias,
        {},
    )

    company_aliases = company_owners.setdefault(
        company_id,
        set(),
    )

    company_aliases.add(
        alias_text.strip()
    )


def build_alias_ownership_index(
    companies: list[dict[str, str]],
    aliases: list[dict[str, str]],
) -> dict[str, dict[str, set[str]]]:
    """
    Normalize edilmiş alias metninin hangi şirketlere ait
    olduğunu gösteren indeks oluşturur.

    Yapı:
        normalized_alias
            -> company_id
                -> alias metinleri
    """

    ownership_index: dict[
        str,
        dict[str, set[str]],
    ] = {}

    for alias_row in aliases:
        company_id = alias_row.get(
            "company_id",
            "",
        ).strip()

        alias_text = alias_row.get(
            "alias_text",
            "",
        ).strip()

        if not company_id or not alias_text:
            continue

        add_alias_owner(
            ownership_index,
            company_id=company_id,
            alias_text=alias_text,
        )

    for company_row in companies:
        company_id = company_row.get(
            "company_id",
            "",
        ).strip()

        if not company_id:
            continue

        for field_name in (
            "legal_name",
            "brand_name",
        ):
            alias_text = company_row.get(
                field_name,
                "",
            ).strip()

            if not alias_text:
                continue

            add_alias_owner(
                ownership_index,
                company_id=company_id,
                alias_text=alias_text,
            )

    return ownership_index


def find_alias_collision(
    *,
    normalized_query: str,
    expected_company_id: str,
    ownership_index: dict[
        str,
        dict[str, set[str]],
    ],
) -> dict[str, Any]:
    owners = ownership_index.get(
        normalized_query,
        {},
    )

    conflicting_company_ids = sorted(
        company_id
        for company_id in owners
        if company_id != expected_company_id
    )

    if not conflicting_company_ids:
        return {
            "has_collision": False,
            "collision_company_ids": [],
            "collision_aliases": [],
            "collision_reason": "",
        }

    conflicting_aliases: list[str] = []

    for company_id in conflicting_company_ids:
        for alias_text in sorted(
            owners.get(
                company_id,
                set(),
            )
        ):
            conflicting_aliases.append(
                f"{company_id}:{alias_text}"
            )

    return {
        "has_collision": True,
        "collision_company_ids": (
            conflicting_company_ids
        ),
        "collision_aliases": (
            conflicting_aliases
        ),
        "collision_reason": (
            COLLISION_REASON
        ),
    }


def generate_validated_variant(
    *,
    expected_company_id: str,
    aliases: list[dict[str, str]],
    variant_index: int,
    difficulty: str,
    rng: random.Random,
    ownership_index: dict[
        str,
        dict[str, set[str]],
    ],
    generated_query_keys: set[str],
    maximum_attempts: int,
) -> dict[str, Any]:
    last_result: dict[str, Any] | None = None

    for attempt_number in range(
        1,
        maximum_attempts + 1,
    ):
        alias_index = (
            variant_index
            + attempt_number
            - 1
        ) % len(aliases)

        alias_row = aliases[
            alias_index
        ]

        source_alias = alias_row[
            "alias_text"
        ].strip()

        query_text, corruption_type = (
            make_variant(
                alias=source_alias,
                difficulty=difficulty,
                rng=rng,
            )
        )

        normalized_query = normalize_text(
            query_text
        )

        query_key = query_text.casefold()

        collision_result = (
            find_alias_collision(
                normalized_query=(
                    normalized_query
                ),
                expected_company_id=(
                    expected_company_id
                ),
                ownership_index=(
                    ownership_index
                ),
            )
        )

        duplicate_query = (
            query_key in generated_query_keys
        )

        benchmark_collision = bool(
            collision_result[
                "has_collision"
            ]
        )

        benchmark_valid = (
            not benchmark_collision
            and not duplicate_query
        )

        result = {
            "source_alias": source_alias,
            "source_alias_type": (
                alias_row.get(
                    "alias_type",
                    "",
                )
            ),
            "query_text": query_text,
            "normalized_query": (
                normalized_query
            ),
            "ambiguous_query": (
                is_ambiguous_query(
                    normalized_query
                )
            ),
            "corruption_type": (
                corruption_type
            ),
            "generation_attempts": (
                attempt_number
            ),
            "duplicate_generated_query": (
                duplicate_query
            ),
            "benchmark_collision": (
                benchmark_collision
            ),
            "benchmark_valid": (
                benchmark_valid
            ),
            "collision_company_ids": (
                "|".join(
                    collision_result[
                        "collision_company_ids"
                    ]
                )
            ),
            "collision_aliases": (
                "|".join(
                    collision_result[
                        "collision_aliases"
                    ][:20]
                )
            ),
            "collision_reason": (
                collision_result[
                    "collision_reason"
                ]
            ),
        }

        last_result = result

        if benchmark_valid:
            generated_query_keys.add(
                query_key
            )

            return result

    if last_result is None:
        raise RuntimeError(
            "Benchmark sorgusu üretilemedi."
        )

    generated_query_keys.add(
        last_result[
            "query_text"
        ].casefold()
    )

    return last_result


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.max_companies < 1:
        raise ValueError(
            "--max-companies en az 1 olmalıdır."
        )

    if arguments.variants_per_company < 1:
        raise ValueError(
            "--variants-per-company en az 1 olmalıdır."
        )

    if arguments.max_regeneration_attempts < 1:
        raise ValueError(
            "--max-regeneration-attempts "
            "en az 1 olmalıdır."
        )


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    rng = random.Random(
        arguments.seed
    )

    companies = read_csv(
        arguments.companies_path
    )

    aliases = read_csv(
        arguments.aliases_path
    )

    ownership_index = (
        build_alias_ownership_index(
            companies=companies,
            aliases=aliases,
        )
    )

    aliases_by_company: dict[
        str,
        list[dict[str, str]],
    ] = defaultdict(list)

    for alias_row in aliases:
        company_id = alias_row.get(
            "company_id",
            "",
        ).strip()

        if company_id:
            aliases_by_company[
                company_id
            ].append(
                alias_row
            )

    eligible_companies = [
        company
        for company in companies
        if aliases_by_company.get(
            company.get(
                "company_id",
                "",
            )
        )
    ]

    rng.shuffle(
        eligible_companies
    )

    selected_companies = (
        eligible_companies[
            :arguments.max_companies
        ]
    )

    difficulties = (
        "EASY",
        "MEDIUM",
        "HARD",
    )

    benchmark_rows: list[
        dict[str, Any]
    ] = []

    benchmark_id = 1

    for company_row in selected_companies:
        company_id = company_row[
            "company_id"
        ].strip()

        company_aliases = choose_aliases(
            aliases_by_company[
                company_id
            ]
        )

        if not company_aliases:
            continue

        generated_query_keys: set[str] = set()

        for variant_index in range(
            arguments.variants_per_company
        ):
            difficulty = difficulties[
                variant_index
                % len(difficulties)
            ]

            generated_variant = (
                generate_validated_variant(
                    expected_company_id=(
                        company_id
                    ),
                    aliases=company_aliases,
                    variant_index=(
                        variant_index
                    ),
                    difficulty=difficulty,
                    rng=rng,
                    ownership_index=(
                        ownership_index
                    ),
                    generated_query_keys=(
                        generated_query_keys
                    ),
                    maximum_attempts=(
                        arguments
                        .max_regeneration_attempts
                    ),
                )
            )

            benchmark_valid = bool(
                generated_variant[
                    "benchmark_valid"
                ]
            )

            ambiguous_query = bool(
                generated_variant[
                    "ambiguous_query"
                ]
            )

            strict_evaluation_eligible = (
                benchmark_valid
                and not ambiguous_query
            )

            benchmark_rows.append(
                {
                    "benchmark_id": (
                        benchmark_id
                    ),
                    "company_id": (
                        company_id
                    ),
                    "legal_name": (
                        company_row.get(
                            "legal_name",
                            "",
                        )
                    ),
                    "brand_name": (
                        company_row.get(
                            "brand_name",
                            "",
                        )
                    ),
                    "country": (
                        company_row.get(
                            "country",
                            "",
                        )
                    ),
                    "source_alias": (
                        generated_variant[
                            "source_alias"
                        ]
                    ),
                    "source_alias_type": (
                        generated_variant[
                            "source_alias_type"
                        ]
                    ),
                    "query_text": (
                        generated_variant[
                            "query_text"
                        ]
                    ),
                    "normalized_query": (
                        generated_variant[
                            "normalized_query"
                        ]
                    ),
                    "difficulty": (
                        difficulty
                    ),
                    "corruption_type": (
                        generated_variant[
                            "corruption_type"
                        ]
                    ),
                    "ambiguous_query": (
                        ambiguous_query
                    ),
                    "benchmark_collision": (
                        generated_variant[
                            "benchmark_collision"
                        ]
                    ),
                    "benchmark_valid": (
                        benchmark_valid
                    ),
                    "strict_evaluation_eligible": (
                        strict_evaluation_eligible
                    ),
                    "collision_company_ids": (
                        generated_variant[
                            "collision_company_ids"
                        ]
                    ),
                    "collision_aliases": (
                        generated_variant[
                            "collision_aliases"
                        ]
                    ),
                    "collision_reason": (
                        generated_variant[
                            "collision_reason"
                        ]
                    ),
                    "duplicate_generated_query": (
                        generated_variant[
                            "duplicate_generated_query"
                        ]
                    ),
                    "generation_attempts": (
                        generated_variant[
                            "generation_attempts"
                        ]
                    ),
                    "seed": (
                        arguments.seed
                    ),
                }
            )

            benchmark_id += 1

    arguments.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "benchmark_id",
        "company_id",
        "legal_name",
        "brand_name",
        "country",
        "source_alias",
        "source_alias_type",
        "query_text",
        "normalized_query",
        "difficulty",
        "corruption_type",
        "ambiguous_query",
        "benchmark_collision",
        "benchmark_valid",
        "strict_evaluation_eligible",
        "collision_company_ids",
        "collision_aliases",
        "collision_reason",
        "duplicate_generated_query",
        "generation_attempts",
        "seed",
    ]

    with arguments.output.open(
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
            benchmark_rows
        )

    difficulty_counts: dict[
        str,
        int,
    ] = defaultdict(int)

    collision_count = 0
    invalid_count = 0
    ambiguous_count = 0
    strict_eligible_count = 0
    regenerated_count = 0

    for benchmark_row in benchmark_rows:
        difficulty_counts[
            str(
                benchmark_row[
                    "difficulty"
                ]
            )
        ] += 1

        if benchmark_row[
            "benchmark_collision"
        ]:
            collision_count += 1

        if not benchmark_row[
            "benchmark_valid"
        ]:
            invalid_count += 1

        if benchmark_row[
            "ambiguous_query"
        ]:
            ambiguous_count += 1

        if benchmark_row[
            "strict_evaluation_eligible"
        ]:
            strict_eligible_count += 1

        if int(
            benchmark_row[
                "generation_attempts"
            ]
        ) > 1:
            regenerated_count += 1

    print()
    print(
        "Benchmark veri seti oluşturuldu."
    )

    print(
        f"Şirket sayısı: "
        f"{len(selected_companies)}"
    )

    print(
        f"Sorgu sayısı: "
        f"{len(benchmark_rows)}"
    )

    print(
        f"Kolay: "
        f"{difficulty_counts['EASY']}"
    )

    print(
        f"Orta: "
        f"{difficulty_counts['MEDIUM']}"
    )

    print(
        f"Zor: "
        f"{difficulty_counts['HARD']}"
    )

    print(
        f"Yeniden üretilen sorgu: "
        f"{regenerated_count}"
    )

    print(
        f"Giderilemeyen alias çakışması: "
        f"{collision_count}"
    )

    print(
        f"Geçersiz benchmark kaydı: "
        f"{invalid_count}"
    )

    print(
        f"Belirsiz sorgu: "
        f"{ambiguous_count}"
    )

    print(
        f"Strict değerlendirmeye uygun: "
        f"{strict_eligible_count}"
    )

    print(
        f"Dosya: {arguments.output}"
    )


if __name__ == "__main__":
    main()