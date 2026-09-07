from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_POSITIVE_INPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "benchmark_queries.csv"
)

DEFAULT_NEGATIVE_INPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "negative_benchmark_queries.csv"
)

DEFAULT_HARD_POSITIVE_INPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "hard_positive_benchmark_queries.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "candidate_feature_dataset.csv"
)

DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "candidate_feature_dataset_summary.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Pozitif, negatif ve hard-positive benchmark "
            "sorgularının tüm aday özelliklerini CatBoostRanker "
            "eğitim veri setine aktarır."
        )
    )

    parser.add_argument(
        "--positive-input",
        type=Path,
        default=DEFAULT_POSITIVE_INPUT,
    )

    parser.add_argument(
        "--negative-input",
        type=Path,
        default=DEFAULT_NEGATIVE_INPUT,
    )

    parser.add_argument(
        "--hard-positive-input",
        type=Path,
        default=DEFAULT_HARD_POSITIVE_INPUT,
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
        "--positive-sample-size",
        type=int,
        default=0,
        help="0 bütün standart pozitif sorguları kullanır.",
    )

    parser.add_argument(
        "--negative-sample-size",
        type=int,
        default=0,
        help="0 bütün negatif sorguları kullanır.",
    )

    parser.add_argument(
        "--hard-positive-sample-size",
        type=int,
        default=0,
        help="0 bütün hard-positive sorguları kullanır.",
    )

    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
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


def validate_arguments(
    arguments: argparse.Namespace,
) -> None:
    if arguments.positive_sample_size < 0:
        raise ValueError(
            "--positive-sample-size negatif olamaz."
        )

    if arguments.negative_sample_size < 0:
        raise ValueError(
            "--negative-sample-size negatif olamaz."
        )

    if arguments.hard_positive_sample_size < 0:
        raise ValueError(
            "--hard-positive-sample-size negatif olamaz."
        )

    if arguments.candidate_limit < 2:
        raise ValueError(
            "--candidate-limit en az 2 olmalıdır."
        )

    if arguments.timeout <= 0:
        raise ValueError(
            "--timeout pozitif olmalıdır."
        )


def read_csv(
    path: Path,
    dataset_name: str,
    *,
    required: bool = True,
) -> list[dict[str, str]]:
    if not path.exists():
        if required:
            raise FileNotFoundError(
                f"{dataset_name} bulunamadı: {path}"
            )

        return []

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    if required and not rows:
        raise ValueError(
            f"{dataset_name} boş."
        )

    return rows


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


def as_integer(
    value: Any,
    default: int = 0,
) -> int:
    if value in (
        None,
        "",
    ):
        return default

    if isinstance(value, bool):
        return int(value)

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


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value in (
        None,
        "",
    ):
        return default

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def normalize_identifier(
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


def filter_valid_rows(
    rows: list[dict[str, str]],
) -> list[dict[str, str]]:
    return [
        row
        for row in rows
        if as_boolean(
            row.get(
                "benchmark_valid",
                True,
            ),
            default=True,
        )
    ]


def sample_rows(
    rows: list[dict[str, str]],
    sample_size: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    if (
        sample_size <= 0
        or sample_size >= len(rows)
    ):
        return list(rows)

    return rng.sample(
        rows,
        sample_size,
    )


def resolve_benchmark_id(
    row: dict[str, str],
    fallback_index: int,
    prefix: str,
) -> str:
    for column_name in (
        "benchmark_id",
        "query_id",
        "id",
    ):
        value = str(
            row.get(
                column_name,
                "",
            )
        ).strip()

        if value:
            return value

    return (
        f"{prefix}-{fallback_index}"
    )


def resolve_expected_company_id(
    row: dict[str, str],
) -> str:
    for column_name in (
        "expected_company_id",
        "company_id",
    ):
        value = normalize_identifier(
            row.get(
                column_name
            )
        )

        if value:
            return value

    return ""


def check_backend(
    session: requests.Session,
    backend_url: str,
    timeout: float,
) -> None:
    response = session.get(
        (
            f"{backend_url.rstrip('/')}"
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
            "Matching endpointi OpenAPI içinde bulunamadı. "
            f"Beklenen yol: {required_path}. "
            f"Mevcut yollar: {sorted(paths)}"
        )


def find_expected_rank(
    candidates: list[dict[str, Any]],
    expected_company_id: str,
) -> int:
    if not expected_company_id:
        return 0

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate_company_id = (
            normalize_identifier(
                candidate.get(
                    "company_id"
                )
            )
        )

        if (
            candidate_company_id
            == expected_company_id
        ):
            return rank

    return 0


def build_candidate_row(
    *,
    dataset_type: str,
    group_id: str,
    benchmark_id: str,
    source_row: dict[str, str],
    payload: dict[str, Any],
    candidate: dict[str, Any],
    candidate_position: int,
    candidate_count: int,
    expected_company_id: str,
    expected_rank: int,
    ranker_eligible: bool,
    latency_ms: float,
) -> dict[str, Any]:
    candidate_company_id = (
        normalize_identifier(
            candidate.get(
                "company_id"
            )
        )
    )

    is_expected_company = (
        bool(expected_company_id)
        and candidate_company_id
        == expected_company_id
    )

    query_length = as_integer(
        candidate.get(
            "query_length",
            payload.get(
                "query_length",
                len(
                    str(
                        source_row.get(
                            "query_text",
                            ""
                        )
                    )
                ),
            ),
        )
    )

    query_token_count = as_integer(
        candidate.get(
            "query_token_count",
            payload.get(
                "query_token_count",
                0,
            ),
        )
    )

    candidate_reranker_gap = as_float(
        candidate.get(
            "candidate_reranker_gap",
            candidate.get(
                "reranker_gap",
                payload.get(
                    "reranker_gap",
                    0.0,
                ),
            ),
        )
    )

    base_reranker_weight = as_float(
        candidate.get(
            "base_reranker_weight",
            payload.get(
                "base_reranker_weight",
                0.0,
            ),
        )
    )

    return {
        # Grup ve benchmark bilgileri
        "dataset_type": dataset_type,
        "group_id": group_id,
        "benchmark_id": benchmark_id,
        "query_text": source_row.get(
            "query_text",
            "",
        ),
        "expected_company_id": expected_company_id,
        "expected_legal_name": source_row.get(
            "expected_legal_name",
            "",
        ),
        "expected_decision": source_row.get(
            "expected_decision",
            (
                "UNKNOWN"
                if dataset_type == "NEGATIVE"
                else ""
            ),
        ),
        "actual_decision": payload.get(
            "decision",
            "",
        ),
        "negative_category": source_row.get(
            "negative_category",
            "",
        ),
        "perturbation_type": source_row.get(
            "perturbation_type",
            "",
        ),
        "perturbation_operations": source_row.get(
            "perturbation_operations",
            "",
        ),
        "source_alias": source_row.get(
            "source_alias",
            "",
        ),
        "source_alias_type": source_row.get(
            "source_alias_type",
            "",
        ),
        "benchmark_valid": 1,
        "ranker_eligible": int(
            ranker_eligible
        ),
        "candidate_count": candidate_count,
        "expected_rank": expected_rank,
        "candidate_position": candidate_position,
        "label": int(
            is_expected_company
        ),
        "is_expected_company": int(
            is_expected_company
        ),

        # Şirket ve alias bilgileri
        "company_id": candidate_company_id,
        "legal_name": candidate.get(
            "legal_name",
            "",
        ),
        "brand_name": candidate.get(
            "brand_name",
            "",
        ),
        "city": candidate.get(
            "city",
            "",
        ),
        "sector": candidate.get(
            "sector",
            "",
        ),
        "matched_alias": candidate.get(
            "matched_alias",
            "",
        ),
        "normalized_alias": candidate.get(
            "normalized_alias",
            "",
        ),
        "alias_type": candidate.get(
            "alias_type",
            "",
        ),

        # Deterministik özellikler
        "exact_match": int(
            as_boolean(
                candidate.get(
                    "exact_match"
                )
            )
        ),
        "unique_exact_match": int(
            as_boolean(
                candidate.get(
                    "unique_exact_match"
                )
            )
        ),
        "ambiguous_exact_match": int(
            as_boolean(
                candidate.get(
                    "ambiguous_exact_match"
                )
            )
        ),
        "identifier_score": as_float(
            candidate.get(
                "identifier_score"
            )
        ),
        "identifier_overlap": int(
            as_boolean(
                candidate.get(
                    "identifier_overlap"
                )
            )
        ),

        # TF-IDF ve lexical özellikler
        "char_wb_tfidf_score": as_float(
            candidate.get(
                "char_wb_tfidf_score"
            )
        ),
        "raw_char_tfidf_score": as_float(
            candidate.get(
                "raw_char_tfidf_score"
            )
        ),
        "char_tfidf_score": as_float(
            candidate.get(
                "char_tfidf_score"
            )
        ),
        "jaro_winkler_score": as_float(
            candidate.get(
                "jaro_winkler_score"
            )
        ),
        "token_prefix_score": as_float(
            candidate.get(
                "token_prefix_score"
            )
        ),
        "fuzzy_ratio_score": as_float(
            candidate.get(
                "fuzzy_ratio_score"
            )
        ),
        "fuzzy_partial_ratio_score": as_float(
            candidate.get(
                "fuzzy_partial_ratio_score"
            )
        ),
        "fuzzy_token_set_score": as_float(
            candidate.get(
                "fuzzy_token_set_score"
            )
        ),
        "fuzzy_wratio_score": as_float(
            candidate.get(
                "fuzzy_wratio_score"
            )
        ),

        # Embedding ve retrieval özellikleri
        "embedding_score": as_float(
            candidate.get(
                "embedding_score"
            )
        ),
        "hybrid_score": as_float(
            candidate.get(
                "hybrid_score"
            )
        ),
        "retrieval_score": as_float(
            candidate.get(
                "retrieval_score"
            )
        ),
        "retrieval_rank": as_integer(
            candidate.get(
                "retrieval_rank",
                candidate_position,
            )
        ),

        # Qwen reranker özellikleri
        "reranker_score": as_float(
            candidate.get(
                "reranker_score"
            )
        ),
        "adjusted_reranker_score": as_float(
            candidate.get(
                "adjusted_reranker_score"
            )
        ),
        "lexical_evidence": as_float(
            candidate.get(
                "lexical_evidence"
            )
        ),
        "retrieval_weight": as_float(
            candidate.get(
                "retrieval_weight"
            )
        ),
        "reranker_weight": as_float(
            candidate.get(
                "reranker_weight"
            )
        ),
        "reranker_gap": as_float(
            candidate.get(
                "reranker_gap",
                payload.get(
                    "reranker_gap",
                    0.0,
                ),
            )
        ),
        "candidate_reranker_gap": (
            candidate_reranker_gap
        ),
        "base_reranker_weight": (
            base_reranker_weight
        ),
        "query_length": query_length,
        "query_token_count": (
            query_token_count
        ),
        "final_score": as_float(
            candidate.get(
                "final_score"
            )
        ),
        "reranked": int(
            as_boolean(
                candidate.get(
                    "reranked"
                )
            )
        ),
        "current_final_rank": as_integer(
            candidate.get(
                "current_final_rank",
                candidate_position,
            )
        ),

        # CatBoost shadow alanları yalnızca analiz içindir.
        # Bir sonraki eğitim scriptinde feature olarak
        # kullanılmayacaktır.
        "catboost_raw_score": as_float(
            candidate.get(
                "catboost_raw_score"
            )
        ),
        "catboost_normalized_score": as_float(
            candidate.get(
                "catboost_normalized_score"
            )
        ),
        "catboost_rank": as_integer(
            candidate.get(
                "catboost_rank"
            )
        ),
        "catboost_shadow_top1": int(
            as_boolean(
                candidate.get(
                    "catboost_shadow_top1"
                )
            )
        ),
        "catboost_shadow_order_delta": (
            as_integer(
                candidate.get(
                    "catboost_shadow_order_delta"
                )
            )
        ),

        # Sorgu seviyesinde tanısal bilgiler
        "ranking_mode": payload.get(
            "ranking_mode",
            "",
        ),
        "reranker_applied": int(
            as_boolean(
                payload.get(
                    "reranker_applied"
                )
            )
        ),
        "catboost_shadow_applied": int(
            as_boolean(
                payload.get(
                    "catboost_shadow_applied"
                )
            )
        ),
        "catboost_shadow_changed_winner": int(
            as_boolean(
                payload.get(
                    "catboost_shadow_changed_winner"
                )
            )
        ),
        "score_margin": as_float(
            payload.get(
                "score_margin"
            )
        ),
        "catboost_shadow_raw_margin": as_float(
            payload.get(
                "catboost_shadow_raw_margin"
            )
        ),
        "latency_ms": round(
            latency_ms,
            3,
        ),
    }


def process_query(
    *,
    session: requests.Session,
    backend_url: str,
    dataset_type: str,
    source_row: dict[str, str],
    fallback_index: int,
    candidate_limit: int,
    timeout: float,
) -> dict[str, Any]:
    benchmark_prefix = {
        "POSITIVE": "POS",
        "NEGATIVE": "NEG",
        "HARD_POSITIVE": "HP",
    }[dataset_type]

    benchmark_id = resolve_benchmark_id(
        source_row,
        fallback_index,
        benchmark_prefix,
    )

    group_id = (
        f"{dataset_type}:{benchmark_id}"
    )

    query_text = str(
        source_row.get(
            "query_text",
            "",
        )
    ).strip()

    expected_company_id = (
        resolve_expected_company_id(
            source_row
        )
        if dataset_type
        in {
            "POSITIVE",
            "HARD_POSITIVE",
        }
        else ""
    )

    endpoint = (
        f"{backend_url.rstrip('/')}"
        "/api/v1/matching/candidates"
    )

    started_at = time.perf_counter()

    try:
        response = session.post(
            endpoint,
            json={
                "text": query_text,
                "limit": candidate_limit,
            },
            timeout=timeout,
        )

        latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000.0

        response.raise_for_status()

        payload = response.json()

        candidates = payload.get(
            "candidates",
            [],
        )

        if not isinstance(
            candidates,
            list,
        ):
            raise TypeError(
                "API candidates alanı liste değil."
            )

        expected_rank = find_expected_rank(
            candidates,
            expected_company_id,
        )

        positive_group = (
            dataset_type
            in {
                "POSITIVE",
                "HARD_POSITIVE",
            }
        )

        ranker_eligible = (
            positive_group
            and expected_rank > 0
            and len(candidates) >= 2
        )

        candidate_rows = [
            build_candidate_row(
                dataset_type=dataset_type,
                group_id=group_id,
                benchmark_id=benchmark_id,
                source_row=source_row,
                payload=payload,
                candidate=candidate,
                candidate_position=position,
                candidate_count=len(
                    candidates
                ),
                expected_company_id=(
                    expected_company_id
                ),
                expected_rank=(
                    expected_rank
                ),
                ranker_eligible=(
                    ranker_eligible
                ),
                latency_ms=latency_ms,
            )
            for position, candidate
            in enumerate(
                candidates,
                start=1,
            )
        ]

        positive_label_count = sum(
            row["label"]
            for row in candidate_rows
        )

        if ranker_eligible:
            if positive_label_count != 1:
                ranker_eligible = False

                for candidate_row in candidate_rows:
                    candidate_row[
                        "ranker_eligible"
                    ] = 0

        return {
            "api_success": True,
            "api_error": "",
            "dataset_type": dataset_type,
            "group_id": group_id,
            "benchmark_id": benchmark_id,
            "query_text": query_text,
            "expected_company_id": (
                expected_company_id
            ),
            "expected_rank": expected_rank,
            "candidate_count": len(
                candidates
            ),
            "positive_label_count": (
                positive_label_count
            ),
            "ranker_eligible": (
                ranker_eligible
            ),
            "actual_decision": payload.get(
                "decision",
                "",
            ),
            "catboost_shadow_changed_winner": (
                as_boolean(
                    payload.get(
                        "catboost_shadow_changed_winner"
                    )
                )
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "candidate_rows": candidate_rows,
        }

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000.0

        return {
            "api_success": False,
            "api_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "dataset_type": dataset_type,
            "group_id": group_id,
            "benchmark_id": benchmark_id,
            "query_text": query_text,
            "expected_company_id": (
                expected_company_id
            ),
            "expected_rank": 0,
            "candidate_count": 0,
            "positive_label_count": 0,
            "ranker_eligible": False,
            "actual_decision": "",
            "catboost_shadow_changed_winner": False,
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "candidate_rows": [],
        }


def write_candidate_rows(
    output_path: Path,
    candidate_rows: list[dict[str, Any]],
) -> None:
    if not candidate_rows:
        raise ValueError(
            "Yazılacak aday satırı bulunamadı."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for row in candidate_rows:
        for field_name in row:
            if field_name not in fieldnames:
                fieldnames.append(
                    field_name
                )

    with output_path.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(
            candidate_rows
        )


def calculate_summary(
    query_results: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    successful_queries = [
        result
        for result in query_results
        if result["api_success"]
    ]

    failed_queries = [
        result
        for result in query_results
        if not result["api_success"]
    ]

    eligible_groups = [
        result
        for result in successful_queries
        if result["ranker_eligible"]
    ]

    positive_groups = [
        result
        for result in successful_queries
        if result["dataset_type"]
        == "POSITIVE"
    ]

    hard_positive_groups = [
        result
        for result in successful_queries
        if result["dataset_type"]
        == "HARD_POSITIVE"
    ]

    negative_groups = [
        result
        for result in successful_queries
        if result["dataset_type"]
        == "NEGATIVE"
    ]

    hard_positive_rank_distribution = Counter(
        result["expected_rank"]
        for result in hard_positive_groups
        if result["expected_rank"] > 0
    )

    dataset_query_counts = Counter(
        result["dataset_type"]
        for result in query_results
    )

    dataset_candidate_counts = Counter(
        row["dataset_type"]
        for row in candidate_rows
    )

    decision_counts = Counter(
        result["actual_decision"]
        for result in successful_queries
    )

    latencies = [
        result["latency_ms"]
        for result in successful_queries
    ]

    hard_positive_changed_winner_count = sum(
        result[
            "catboost_shadow_changed_winner"
        ]
        for result in hard_positive_groups
    )

    positive_recall_count = sum(
        result["expected_rank"] > 0
        for result in positive_groups
    )

    hard_positive_recall_count = sum(
        result["expected_rank"] > 0
        for result in hard_positive_groups
    )

    eligible_label_counts = Counter(
        result["positive_label_count"]
        for result in eligible_groups
    )

    return {
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "total_queries": len(
            query_results
        ),
        "successful_queries": len(
            successful_queries
        ),
        "failed_queries": len(
            failed_queries
        ),
        "query_counts_by_dataset": dict(
            dataset_query_counts
        ),
        "total_candidate_rows": len(
            candidate_rows
        ),
        "candidate_rows_by_dataset": dict(
            dataset_candidate_counts
        ),
        "ranker_eligible_groups": len(
            eligible_groups
        ),
        "ranker_ineligible_groups": (
            len(successful_queries)
            - len(eligible_groups)
        ),
        "eligible_group_label_counts": dict(
            eligible_label_counts
        ),
        "standard_positive_groups": len(
            positive_groups
        ),
        "standard_positive_recall_at_k": (
            positive_recall_count
            / len(positive_groups)
            if positive_groups
            else None
        ),
        "hard_positive_groups": len(
            hard_positive_groups
        ),
        "hard_positive_recall_at_k": (
            hard_positive_recall_count
            / len(hard_positive_groups)
            if hard_positive_groups
            else None
        ),
        "hard_positive_rank_distribution": {
            str(rank): count
            for rank, count
            in sorted(
                hard_positive_rank_distribution.items()
            )
        },
        "hard_positive_catboost_changed_winner_count": (
            hard_positive_changed_winner_count
        ),
        "negative_groups": len(
            negative_groups
        ),
        "decision_counts": dict(
            decision_counts
        ),
        "average_latency_ms": (
            round(
                statistics.mean(
                    latencies
                ),
                3,
            )
            if latencies
            else None
        ),
        "median_latency_ms": (
            round(
                statistics.median(
                    latencies
                ),
                3,
            )
            if latencies
            else None
        ),
        "failed_query_details": [
            {
                "dataset_type": result[
                    "dataset_type"
                ],
                "benchmark_id": result[
                    "benchmark_id"
                ],
                "query_text": result[
                    "query_text"
                ],
                "api_error": result[
                    "api_error"
                ],
            }
            for result in failed_queries
        ],
    }


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    positive_rows = filter_valid_rows(
        read_csv(
            arguments.positive_input,
            "Pozitif benchmark",
        )
    )

    negative_rows = filter_valid_rows(
        read_csv(
            arguments.negative_input,
            "Negatif benchmark",
        )
    )

    hard_positive_rows = filter_valid_rows(
        read_csv(
            arguments.hard_positive_input,
            "Hard-positive benchmark",
        )
    )

    rng = random.Random(
        arguments.seed
    )

    positive_rows = sample_rows(
        positive_rows,
        arguments.positive_sample_size,
        rng,
    )

    negative_rows = sample_rows(
        negative_rows,
        arguments.negative_sample_size,
        rng,
    )

    hard_positive_rows = sample_rows(
        hard_positive_rows,
        arguments.hard_positive_sample_size,
        rng,
    )

    work_items: list[
        tuple[
            str,
            dict[str, str],
            int,
        ]
    ] = []

    work_items.extend(
        (
            "POSITIVE",
            row,
            index,
        )
        for index, row in enumerate(
            positive_rows,
            start=1,
        )
    )

    work_items.extend(
        (
            "NEGATIVE",
            row,
            index,
        )
        for index, row in enumerate(
            negative_rows,
            start=1,
        )
    )

    work_items.extend(
        (
            "HARD_POSITIVE",
            row,
            index,
        )
        for index, row in enumerate(
            hard_positive_rows,
            start=1,
        )
    )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "eft-candidate-feature-exporter/2.0"
            ),
        }
    )

    query_results: list[
        dict[str, Any]
    ] = []

    candidate_rows: list[
        dict[str, Any]
    ] = []

    try:
        check_backend(
            session=session,
            backend_url=(
                arguments.backend_url
            ),
            timeout=arguments.timeout,
        )

        print(
            "Backend ve matching endpointi hazır."
        )

        print(
            "Standart pozitif sorgu: "
            f"{len(positive_rows)}"
        )

        print(
            "Negatif sorgu: "
            f"{len(negative_rows)}"
        )

        print(
            "Hard-positive sorgu: "
            f"{len(hard_positive_rows)}"
        )

        total_count = len(
            work_items
        )

        for work_index, (
            dataset_type,
            source_row,
            fallback_index,
        ) in enumerate(
            work_items,
            start=1,
        ):
            query_result = process_query(
                session=session,
                backend_url=(
                    arguments.backend_url
                ),
                dataset_type=dataset_type,
                source_row=source_row,
                fallback_index=(
                    fallback_index
                ),
                candidate_limit=(
                    arguments.candidate_limit
                ),
                timeout=arguments.timeout,
            )

            query_results.append(
                query_result
            )

            candidate_rows.extend(
                query_result[
                    "candidate_rows"
                ]
            )

            if (
                work_index == 1
                or work_index % 25 == 0
                or work_index == total_count
            ):
                successful_count = sum(
                    result["api_success"]
                    for result in query_results
                )

                eligible_count = sum(
                    result["ranker_eligible"]
                    for result in query_results
                )

                hard_eligible_count = sum(
                    (
                        result["dataset_type"]
                        == "HARD_POSITIVE"
                        and result[
                            "ranker_eligible"
                        ]
                    )
                    for result in query_results
                )

                print(
                    f"{work_index}/{total_count} tamamlandı. "
                    f"Başarılı: {successful_count} | "
                    f"Ranker uygun grup: {eligible_count} | "
                    f"Hard-positive uygun: "
                    f"{hard_eligible_count} | "
                    f"Satır: {len(candidate_rows)}"
                )

    finally:
        session.close()

    write_candidate_rows(
        arguments.output,
        candidate_rows,
    )

    summary = calculate_summary(
        query_results,
        candidate_rows,
    )

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
        "Candidate feature veri seti oluşturuldu."
    )

    print(
        "Toplam sorgu: "
        f"{summary['total_queries']}"
    )

    print(
        "Başarılı sorgu: "
        f"{summary['successful_queries']}"
    )

    print(
        "Başarısız sorgu: "
        f"{summary['failed_queries']}"
    )

    print(
        "Toplam aday satırı: "
        f"{summary['total_candidate_rows']}"
    )

    print(
        "Ranker'a uygun grup: "
        f"{summary['ranker_eligible_groups']}"
    )

    print(
        "Standart pozitif Recall@K: "
        + (
            f"{summary['standard_positive_recall_at_k']:.2%}"
            if summary[
                "standard_positive_recall_at_k"
            ] is not None
            else "N/A"
        )
    )

    print(
        "Hard-positive Recall@K: "
        + (
            f"{summary['hard_positive_recall_at_k']:.2%}"
            if summary[
                "hard_positive_recall_at_k"
            ] is not None
            else "N/A"
        )
    )

    print(
        "Hard-positive sıra dağılımı: "
        f"{summary['hard_positive_rank_distribution']}"
    )

    print(
        "Hard-positive CatBoost winner değişimi: "
        f"{summary['hard_positive_catboost_changed_winner_count']}"
    )

    print(
        f"Veri seti: {arguments.output}"
    )

    print(
        f"Özet: {arguments.summary_output}"
    )


if __name__ == "__main__":
    main()