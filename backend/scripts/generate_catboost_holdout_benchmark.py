from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests

from generate_hard_positive_benchmark import (
    PIPELINES,
    build_row,
    build_sources,
    check_backend,
    find_expected,
    identifier,
    mutate,
    normalize,
    read_csv,
    write_csv,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CANDIDATE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "candidate_feature_dataset.csv"
)

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

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "hard_positive_holdout_queries.csv"
)

DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "hard_positive_holdout_summary.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CatBoost eğitim veri setinde bulunmayan "
            "şirketlerden company-disjoint hard-positive "
            "holdout benchmark üretir."
        )
    )

    parser.add_argument(
        "--candidate-dataset",
        type=Path,
        default=DEFAULT_CANDIDATE_DATASET,
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
        default=DEFAULT_OUTPUT,
    )

    parser.add_argument(
        "--summary-output",
        type=Path,
        default=DEFAULT_SUMMARY_OUTPUT,
    )

    parser.add_argument(
        "--backend-url",
        type=str,
        default="http://127.0.0.1:8000",
    )

    parser.add_argument(
        "--target-count",
        type=int,
        default=100,
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
        default=30000,
    )

    parser.add_argument(
        "--max-per-company",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260729,
    )

    return parser.parse_args()


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.target_count < 1:
        raise ValueError(
            "--target-count en az 1 olmalıdır."
        )

    if arguments.candidate_limit < 2:
        raise ValueError(
            "--candidate-limit en az 2 olmalıdır."
        )

    if arguments.minimum_expected_rank < 2:
        raise ValueError(
            "--minimum-expected-rank en az 2 olmalıdır."
        )

    if (
        arguments.maximum_expected_rank
        < arguments.minimum_expected_rank
    ):
        raise ValueError(
            "Maksimum beklenen sıra minimum sıradan "
            "küçük olamaz."
        )

    if (
        arguments.maximum_expected_rank
        > arguments.candidate_limit
    ):
        raise ValueError(
            "Maksimum beklenen sıra candidate-limit "
            "değerini aşamaz."
        )

    if arguments.max_attempts < arguments.target_count:
        raise ValueError(
            "--max-attempts hedef sorgu sayısından "
            "küçük olamaz."
        )

    if arguments.max_per_company < 1:
        raise ValueError(
            "--max-per-company en az 1 olmalıdır."
        )

    if arguments.timeout <= 0:
        raise ValueError(
            "--timeout pozitif olmalıdır."
        )


def as_boolean(
    value: Any,
    default: bool = False,
) -> bool:
    if value in (
        None,
        "",
    ):
        return default

    if isinstance(value, bool):
        return value

    return str(value).strip().casefold() in {
        "1",
        "true",
        "yes",
        "evet",
    }


def read_seen_company_ids(
    candidate_dataset_path: Path,
) -> set[str]:
    """
    CatBoost eğitimine uygun bütün gruplardaki beklenen
    şirket kimliklerini döndürür.

    Bu şirketlerin hiçbiri holdout veri setine alınmaz.
    """

    if not candidate_dataset_path.exists():
        raise FileNotFoundError(
            "Candidate feature veri seti bulunamadı: "
            f"{candidate_dataset_path}"
        )

    with candidate_dataset_path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        reader = csv.DictReader(
            csv_file
        )

        fieldnames = set(
            reader.fieldnames
            or []
        )

        required_columns = {
            "ranker_eligible",
            "expected_company_id",
        }

        missing_columns = (
            required_columns
            - fieldnames
        )

        if missing_columns:
            raise ValueError(
                "Candidate feature veri setinde gerekli "
                f"kolonlar eksik: {sorted(missing_columns)}"
            )

        seen_company_ids: set[str] = set()

        for row in reader:
            if not as_boolean(
                row.get(
                    "ranker_eligible"
                )
            ):
                continue

            company_id = identifier(
                row.get(
                    "expected_company_id"
                )
            )

            if company_id:
                seen_company_ids.add(
                    company_id
                )

    if not seen_company_ids:
        raise ValueError(
            "Candidate feature veri setinden eğitim "
            "şirketleri çıkarılamadı."
        )

    return seen_company_ids


def safe_response_payload(
    response: requests.Response,
) -> dict[str, Any]:
    if not response.ok:
        response_body = response.text.strip()

        if len(response_body) > 1500:
            response_body = (
                response_body[:1500]
                + "..."
            )

        raise RuntimeError(
            f"HTTP {response.status_code}: "
            f"{response_body}"
        )

    payload = response.json()

    if not isinstance(
        payload,
        dict,
    ):
        raise TypeError(
            "API cevabının kök değeri sözlük değil."
        )

    return payload


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    seen_company_ids = read_seen_company_ids(
        arguments.candidate_dataset
    )

    company_rows = read_csv(
        arguments.companies_path,
        "Şirket CSV",
    )

    alias_rows = read_csv(
        arguments.aliases_path,
        "Alias CSV",
    )

    (
        all_sources,
        alias_owners,
    ) = build_sources(
        company_rows,
        alias_rows,
    )

    holdout_sources = [
        source
        for source in all_sources
        if source.company_id
        not in seen_company_ids
    ]

    holdout_company_ids = {
        source.company_id
        for source in holdout_sources
    }

    if not holdout_sources:
        raise ValueError(
            "Eğitim şirketleri çıkarıldıktan sonra "
            "holdout alias kaynağı kalmadı."
        )

    print(
        "Backend kontrolünden önce veri ayrımı hazır."
    )

    print(
        "Eğitimde görülen şirket: "
        f"{len(seen_company_ids)}"
    )

    print(
        "Holdout için kullanılabilir şirket: "
        f"{len(holdout_company_ids)}"
    )

    print(
        "Holdout için kullanılabilir alias: "
        f"{len(holdout_sources)}"
    )

    random_generator = random.Random(
        arguments.seed
    )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "eft-catboost-company-disjoint-holdout/1.0"
            ),
        }
    )

    accepted_rows: list[
        dict[str, Any]
    ] = []

    seen_queries: set[str] = set()

    accepted_company_counts: Counter[
        str
    ] = Counter()

    accepted_pipeline_counts: Counter[
        str
    ] = Counter()

    rejection_counts: Counter[
        str
    ] = Counter()

    try:
        check_backend(
            session,
            arguments.backend_url,
            arguments.timeout,
        )

        print(
            "Backend ve matching endpointi hazır."
        )

        print(
            "Hedef holdout sorgu: "
            f"{arguments.target_count}"
        )

        for attempt in range(
            1,
            arguments.max_attempts + 1,
        ):
            if (
                len(accepted_rows)
                >= arguments.target_count
            ):
                break

            source = random_generator.choice(
                holdout_sources
            )

            if (
                source.company_id
                in seen_company_ids
            ):
                raise RuntimeError(
                    "Company-disjoint ihlali: eğitimde "
                    "görülen şirket holdout kaynağına girdi."
                )

            if (
                accepted_company_counts[
                    source.company_id
                ]
                >= arguments.max_per_company
            ):
                rejection_counts[
                    "COMPANY_LIMIT"
                ] += 1

                continue

            pipeline = random_generator.choice(
                PIPELINES
            )

            query_text = mutate(
                source,
                pipeline,
                random_generator,
            )

            normalized_query = (
                normalize(
                    query_text
                )
                if query_text
                else ""
            )

            if not normalized_query:
                rejection_counts[
                    "EMPTY"
                ] += 1

                continue

            if (
                normalized_query.casefold()
                == source.normalized_alias.casefold()
            ):
                rejection_counts[
                    "UNCHANGED"
                ] += 1

                continue

            query_key = (
                normalized_query.casefold()
            )

            if query_key in seen_queries:
                rejection_counts[
                    "DUPLICATE"
                ] += 1

                continue

            collision_owners = alias_owners.get(
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
                rejection_counts[
                    "ALIAS_COLLISION"
                ] += 1

                continue

            endpoint = (
                f"{arguments.backend_url.rstrip('/')}"
                "/api/v1/matching/candidates"
            )

            try:
                response = session.post(
                    endpoint,
                    json={
                        "text": query_text,
                        "limit": (
                            arguments.candidate_limit
                        ),
                    },
                    timeout=arguments.timeout,
                )

                payload = safe_response_payload(
                    response
                )

            except Exception:
                rejection_counts[
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
                rejection_counts[
                    "INVALID_CANDIDATES"
                ] += 1

                continue

            (
                expected_rank,
                expected_candidate,
            ) = find_expected(
                candidates,
                source.company_id,
            )

            if expected_rank == 0:
                rejection_counts[
                    "EXPECTED_MISSING"
                ] += 1

                continue

            if (
                expected_rank
                < arguments.minimum_expected_rank
            ):
                rejection_counts[
                    "TOO_EASY"
                ] += 1

                continue

            if (
                expected_rank
                > arguments.maximum_expected_rank
            ):
                rejection_counts[
                    "RANK_TOO_LOW"
                ] += 1

                continue

            if expected_candidate is None:
                rejection_counts[
                    "EXPECTED_NONE"
                ] += 1

                continue

            generated_row = build_row(
                len(accepted_rows) + 1,
                source,
                pipeline,
                query_text,
                normalized_query,
                payload,
                expected_rank,
                expected_candidate,
                attempt,
                arguments.seed,
            )

            generated_row.update(
                {
                    "benchmark_id": (
                        f"HOLDOUT-HP-"
                        f"{len(accepted_rows) + 1}"
                    ),
                    "benchmark_type": (
                        "HARD_POSITIVE_HOLDOUT"
                    ),
                    "benchmark_split": (
                        "COMPANY_DISJOINT_HOLDOUT"
                    ),
                    "company_unseen_in_training": True,
                    "excluded_training_company_count": (
                        len(
                            seen_company_ids
                        )
                    ),
                    "holdout_company_pool_count": (
                        len(
                            holdout_company_ids
                        )
                    ),
                    "candidate_dataset_path": str(
                        arguments.candidate_dataset.resolve()
                    ),
                    "generation_created_at": (
                        datetime.now(
                            UTC
                        ).isoformat()
                    ),
                }
            )

            seen_queries.add(
                query_key
            )

            accepted_company_counts[
                source.company_id
            ] += 1

            accepted_pipeline_counts[
                pipeline.name
            ] += 1

            accepted_rows.append(
                generated_row
            )

            if (
                len(accepted_rows) == 1
                or len(accepted_rows) % 10 == 0
            ):
                print(
                    f"Kabul: {len(accepted_rows)}/"
                    f"{arguments.target_count} | "
                    f"Deneme: {attempt} | "
                    f"Beklenen sıra: {expected_rank}"
                )

            if attempt % 1000 == 0:
                print(
                    f"Deneme: {attempt} | "
                    f"Kabul: {len(accepted_rows)} | "
                    "Too easy: "
                    f"{rejection_counts['TOO_EASY']} | "
                    "Expected missing: "
                    f"{rejection_counts['EXPECTED_MISSING']}"
                )

    finally:
        session.close()

    if not accepted_rows:
        raise RuntimeError(
            "Hiç company-disjoint holdout sorgusu "
            "üretilemedi."
        )

    accepted_company_ids = {
        identifier(
            row.get(
                "expected_company_id"
            )
        )
        for row in accepted_rows
    }

    company_overlap = (
        accepted_company_ids
        & seen_company_ids
    )

    if company_overlap:
        raise RuntimeError(
            "Holdout ile eğitim şirketleri çakışıyor: "
            f"{sorted(company_overlap)}"
        )

    write_csv(
        arguments.output,
        accepted_rows,
    )

    rank_distribution = Counter(
        int(
            row[
                "expected_rank_before_catboost"
            ]
        )
        for row in accepted_rows
    )

    catboost_improved_count = sum(
        int(
            row[
                "expected_rank_after_catboost"
            ]
        )
        > 0
        and int(
            row[
                "expected_rank_after_catboost"
            ]
        )
        < int(
            row[
                "expected_rank_before_catboost"
            ]
        )
        for row in accepted_rows
    )

    catboost_hurt_count = sum(
        int(
            row[
                "expected_rank_after_catboost"
            ]
        )
        > int(
            row[
                "expected_rank_before_catboost"
            ]
        )
        for row in accepted_rows
    )

    summary = {
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "split_type": (
            "COMPANY_DISJOINT_HOLDOUT"
        ),
        "candidate_dataset": str(
            arguments.candidate_dataset.resolve()
        ),
        "excluded_training_company_count": len(
            seen_company_ids
        ),
        "available_holdout_company_count": len(
            holdout_company_ids
        ),
        "generated_query_count": len(
            accepted_rows
        ),
        "generated_company_count": len(
            accepted_company_ids
        ),
        "company_overlap_count": len(
            company_overlap
        ),
        "rank_distribution": {
            str(rank): count
            for rank, count
            in sorted(
                rank_distribution.items()
            )
        },
        "catboost_improved_during_generation": (
            catboost_improved_count
        ),
        "catboost_hurt_during_generation": (
            catboost_hurt_count
        ),
        "pipeline_distribution": dict(
            accepted_pipeline_counts
        ),
        "rejection_counts": dict(
            rejection_counts
        ),
        "seed": arguments.seed,
        "maximum_per_company": (
            arguments.max_per_company
        ),
        "output": str(
            arguments.output.resolve()
        ),
    }

    arguments.summary_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.summary_output.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Company-disjoint holdout benchmark oluşturuldu."
    )

    print(
        "Üretilen sorgu: "
        f"{len(accepted_rows)}"
    )

    print(
        "Farklı holdout şirketi: "
        f"{len(accepted_company_ids)}"
    )

    print(
        "Eğitim-holdout şirket çakışması: "
        f"{len(company_overlap)}"
    )

    print(
        "Beklenen sıra dağılımı: "
        f"{dict(sorted(rank_distribution.items()))}"
    )

    print(
        "Üretim sırasında CatBoost iyileştirdi: "
        f"{catboost_improved_count}"
    )

    print(
        "Üretim sırasında CatBoost kötüleştirdi: "
        f"{catboost_hurt_count}"
    )

    print(
        "Ret nedenleri: "
        f"{dict(rejection_counts)}"
    )

    print(
        f"Holdout dosyası: {arguments.output}"
    )

    print(
        f"Özet dosyası: {arguments.summary_output}"
    )

    if (
        len(accepted_rows)
        < arguments.target_count
    ):
        print(
            "Uyarı: Hedef sayıya ulaşılamadı. "
            "--max-attempts artırılabilir."
        )


if __name__ == "__main__":
    main()