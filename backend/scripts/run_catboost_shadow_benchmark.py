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

DEFAULT_RESULTS_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "catboost_shadow_full_benchmark_results.csv"
)

DEFAULT_SUMMARY_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "catboost_shadow_full_benchmark_summary.json"
)


RANKED_DATASET_TYPES = {
    "POSITIVE",
    "HARD_POSITIVE",
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standart pozitif, negatif ve hard-positive "
            "benchmark sorgularında mevcut sıralama ile "
            "CatBoost shadow sıralamasını karşılaştırır."
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
        "--results-output",
        type=Path,
        default=DEFAULT_RESULTS_OUTPUT,
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
        "--candidate-limit",
        type=int,
        default=10,
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
        "--expected-model-path",
        type=Path,
        default=None,
        help=(
            "Backend runtime endpointinde yüklenen CatBoost "
            "model yolunu doğrular."
        ),
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
    if arguments.candidate_limit < 2:
        raise ValueError(
            "--candidate-limit en az 2 olmalıdır."
        )

    sample_sizes = {
        "--positive-sample-size": (
            arguments.positive_sample_size
        ),
        "--negative-sample-size": (
            arguments.negative_sample_size
        ),
        "--hard-positive-sample-size": (
            arguments.hard_positive_sample_size
        ),
    }

    for argument_name, value in sample_sizes.items():
        if value < 0:
            raise ValueError(
                f"{argument_name} negatif olamaz."
            )

    if arguments.timeout <= 0:
        raise ValueError(
            "--timeout pozitif olmalıdır."
        )


def read_csv(
    path: Path,
    dataset_name: str,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"{dataset_name} bulunamadı: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        rows = list(
            csv.DictReader(csv_file)
        )

    if not rows:
        raise ValueError(
            f"{dataset_name} veri seti boş."
        )

    if "query_text" not in rows[0]:
        raise ValueError(
            f"{dataset_name} içinde query_text "
            "kolonu bulunamadı."
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
        "true",
        "1",
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
    random_generator: random.Random,
) -> list[dict[str, str]]:
    if (
        sample_size <= 0
        or sample_size >= len(rows)
    ):
        return list(rows)

    return random_generator.sample(
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

    return f"{prefix}-{fallback_index}"


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


def safe_response_json(
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
            "Backend cevabının kök değeri sözlük değil."
        )

    return payload


def check_backend(
    session: requests.Session,
    backend_url: str,
    timeout: float,
    expected_model_path: Path | None,
) -> dict[str, Any] | None:
    normalized_backend_url = (
        backend_url.rstrip("/")
    )

    openapi_response = session.get(
        f"{normalized_backend_url}/openapi.json",
        timeout=timeout,
    )

    openapi_payload = safe_response_json(
        openapi_response
    )

    available_paths = openapi_payload.get(
        "paths",
        {},
    )

    required_path = (
        "/api/v1/matching/candidates"
    )

    if required_path not in available_paths:
        raise RuntimeError(
            "Matching endpointi OpenAPI içinde bulunamadı. "
            f"Beklenen: {required_path}. "
            f"Mevcut yollar: {sorted(available_paths)}"
        )

    runtime_response = session.get(
        f"{normalized_backend_url}/api/v1/runtime",
        timeout=timeout,
    )

    if runtime_response.status_code == 404:
        return None

    runtime_payload = safe_response_json(
        runtime_response
    )

    catboost_status = runtime_payload.get(
        "catboost_shadow",
        {},
    )

    if not isinstance(
        catboost_status,
        dict,
    ):
        catboost_status = {}

    if (
        catboost_status
        and not as_boolean(
            catboost_status.get(
                "loaded"
            )
        )
    ):
        raise RuntimeError(
            "Backend CatBoost shadow modeli yüklü değil. "
            f"Durum: {catboost_status}"
        )

    if expected_model_path is not None:
        actual_model_path = str(
            catboost_status.get(
                "model_path",
                "",
            )
        ).strip()

        if not actual_model_path:
            raise RuntimeError(
                "Runtime cevabında CatBoost model yolu "
                "bulunamadı."
            )

        expected_resolved = (
            expected_model_path
            if expected_model_path.is_absolute()
            else (
                PROJECT_ROOT
                / expected_model_path
            )
        ).resolve()

        actual_resolved = Path(
            actual_model_path
        ).resolve()

        if (
            actual_resolved
            != expected_resolved
        ):
            raise RuntimeError(
                "Backend beklenen CatBoost modelini "
                "kullanmıyor. "
                f"Beklenen: {expected_resolved}, "
                f"Gerçek: {actual_resolved}"
            )

    return runtime_payload


def find_current_rank(
    candidates: list[dict[str, Any]],
    expected_company_id: str,
) -> int:
    if not expected_company_id:
        return 0

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        if (
            normalize_identifier(
                candidate.get(
                    "company_id"
                )
            )
            == expected_company_id
        ):
            return rank

    return 0


def find_shadow_rank(
    candidates: list[dict[str, Any]],
    expected_company_id: str,
) -> int:
    if not expected_company_id:
        return 0

    for candidate in candidates:
        if (
            normalize_identifier(
                candidate.get(
                    "company_id"
                )
            )
            != expected_company_id
        ):
            continue

        return as_integer(
            candidate.get(
                "catboost_rank"
            )
        )

    return 0


def find_shadow_winner(
    candidates: list[dict[str, Any]],
) -> dict[str, Any] | None:
    ranked_candidates = [
        candidate
        for candidate in candidates
        if as_integer(
            candidate.get(
                "catboost_rank"
            )
        ) > 0
    ]

    if not ranked_candidates:
        return None

    return min(
        ranked_candidates,
        key=lambda candidate: as_integer(
            candidate.get(
                "catboost_rank"
            ),
            default=999999,
        ),
    )


def classify_rank_change(
    current_rank: int,
    shadow_rank: int,
) -> str:
    if current_rank <= 0:
        return (
            "EXPECTED_NOT_IN_CURRENT_CANDIDATES"
        )

    if shadow_rank <= 0:
        return (
            "EXPECTED_NOT_IN_SHADOW_RANKING"
        )

    if shadow_rank < current_rank:
        return "IMPROVED"

    if shadow_rank > current_rank:
        return "HURT"

    return "UNCHANGED"


def reciprocal_rank(
    rank: int,
) -> float:
    if rank <= 0:
        return 0.0

    return 1.0 / rank


def ndcg_single_relevant(
    rank: int,
) -> float:
    if rank <= 0:
        return 0.0

    import math

    return 1.0 / math.log2(
        rank + 1
    )


def serialize_candidates(
    candidates: list[dict[str, Any]],
    *,
    ordering: str,
    limit: int = 5,
) -> str:
    if ordering == "CATBOOST":
        ordered_candidates = sorted(
            candidates,
            key=lambda candidate: (
                as_integer(
                    candidate.get(
                        "catboost_rank"
                    ),
                    default=999999,
                )
                or 999999
            ),
        )

    else:
        ordered_candidates = list(
            candidates
        )

    serialized_candidates: list[
        dict[str, Any]
    ] = []

    for current_position, candidate in enumerate(
        ordered_candidates[:limit],
        start=1,
    ):
        serialized_candidates.append(
            {
                "position": current_position,
                "company_id": candidate.get(
                    "company_id"
                ),
                "legal_name": candidate.get(
                    "legal_name",
                    "",
                ),
                "current_final_rank": (
                    candidate.get(
                        "current_final_rank",
                        "",
                    )
                ),
                "catboost_rank": (
                    candidate.get(
                        "catboost_rank",
                        "",
                    )
                ),
                "retrieval_score": (
                    candidate.get(
                        "retrieval_score",
                        "",
                    )
                ),
                "reranker_score": (
                    candidate.get(
                        "reranker_score",
                        "",
                    )
                ),
                "final_score": (
                    candidate.get(
                        "final_score",
                        "",
                    )
                ),
                "catboost_raw_score": (
                    candidate.get(
                        "catboost_raw_score",
                        "",
                    )
                ),
            }
        )

    return json.dumps(
        serialized_candidates,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


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
    prefixes = {
        "POSITIVE": "POS",
        "NEGATIVE": "NEG",
        "HARD_POSITIVE": "HP",
    }

    benchmark_id = resolve_benchmark_id(
        source_row,
        fallback_index,
        prefixes[dataset_type],
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
        in RANKED_DATASET_TYPES
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

        payload = safe_response_json(
            response
        )

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

        current_winner = (
            candidates[0]
            if candidates
            else None
        )

        shadow_winner = find_shadow_winner(
            candidates
        )

        current_rank = find_current_rank(
            candidates,
            expected_company_id,
        )

        shadow_rank = find_shadow_rank(
            candidates,
            expected_company_id,
        )

        current_winner_company_id = (
            normalize_identifier(
                current_winner.get(
                    "company_id"
                )
            )
            if current_winner
            else ""
        )

        shadow_winner_company_id = (
            normalize_identifier(
                shadow_winner.get(
                    "company_id"
                )
            )
            if shadow_winner
            else ""
        )

        computed_winner_changed = (
            bool(
                current_winner_company_id
            )
            and bool(
                shadow_winner_company_id
            )
            and current_winner_company_id
            != shadow_winner_company_id
        )

        rank_change = (
            classify_rank_change(
                current_rank,
                shadow_rank,
            )
            if dataset_type
            in RANKED_DATASET_TYPES
            else "NOT_APPLICABLE"
        )

        return {
            "dataset_type": dataset_type,
            "benchmark_id": benchmark_id,
            "negative_category": source_row.get(
                "negative_category",
                "",
            ),
            "perturbation_type": source_row.get(
                "perturbation_type",
                "",
            ),
            "query_text": query_text,
            "expected_company_id": (
                expected_company_id
            ),
            "source_expected_rank_before_catboost": (
                source_row.get(
                    "expected_rank_before_catboost",
                    "",
                )
            ),
            "source_expected_rank_after_catboost": (
                source_row.get(
                    "expected_rank_after_catboost",
                    "",
                )
            ),
            "api_success": True,
            "api_error": "",
            "http_status": (
                response.status_code
            ),
            "actual_decision": payload.get(
                "decision",
                "",
            ),
            "ranking_mode": payload.get(
                "ranking_mode",
                "",
            ),
            "candidate_count": len(
                candidates
            ),
            "reranker_applied": as_boolean(
                payload.get(
                    "reranker_applied"
                )
            ),
            "catboost_shadow_status": payload.get(
                "catboost_shadow_status",
                "",
            ),
            "catboost_shadow_applied": as_boolean(
                payload.get(
                    "catboost_shadow_applied"
                )
            ),
            "catboost_shadow_error": payload.get(
                "catboost_shadow_error",
                "",
            ),
            "api_changed_winner": as_boolean(
                payload.get(
                    "catboost_shadow_changed_winner"
                )
            ),
            "computed_changed_winner": (
                computed_winner_changed
            ),
            "winner_change_flag_matches": (
                as_boolean(
                    payload.get(
                        "catboost_shadow_changed_winner"
                    )
                )
                == computed_winner_changed
            ),
            "current_winner_company_id": (
                current_winner_company_id
            ),
            "current_winner_legal_name": (
                current_winner.get(
                    "legal_name",
                    "",
                )
                if current_winner
                else ""
            ),
            "shadow_winner_company_id": (
                shadow_winner_company_id
            ),
            "shadow_winner_legal_name": (
                shadow_winner.get(
                    "legal_name",
                    "",
                )
                if shadow_winner
                else ""
            ),
            "current_expected_rank": (
                current_rank
            ),
            "shadow_expected_rank": (
                shadow_rank
            ),
            "rank_change": rank_change,
            "current_top1_correct": (
                bool(
                    expected_company_id
                )
                and current_rank == 1
            ),
            "shadow_top1_correct": (
                bool(
                    expected_company_id
                )
                and shadow_rank == 1
            ),
            "current_recall_at_k": (
                bool(
                    expected_company_id
                )
                and current_rank > 0
            ),
            "shadow_recall_at_k": (
                bool(
                    expected_company_id
                )
                and shadow_rank > 0
            ),
            "current_reciprocal_rank": (
                reciprocal_rank(
                    current_rank
                )
            ),
            "shadow_reciprocal_rank": (
                reciprocal_rank(
                    shadow_rank
                )
            ),
            "current_ndcg": (
                ndcg_single_relevant(
                    current_rank
                )
            ),
            "shadow_ndcg": (
                ndcg_single_relevant(
                    shadow_rank
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
            "catboost_shadow_normalized_margin": (
                as_float(
                    payload.get(
                        "catboost_shadow_normalized_margin"
                    )
                )
            ),
            "current_top5_json": (
                serialize_candidates(
                    candidates,
                    ordering="CURRENT",
                )
            ),
            "shadow_top5_json": (
                serialize_candidates(
                    candidates,
                    ordering="CATBOOST",
                )
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
        }

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000.0

        return {
            "dataset_type": dataset_type,
            "benchmark_id": benchmark_id,
            "negative_category": source_row.get(
                "negative_category",
                "",
            ),
            "perturbation_type": source_row.get(
                "perturbation_type",
                "",
            ),
            "query_text": query_text,
            "expected_company_id": (
                expected_company_id
            ),
            "source_expected_rank_before_catboost": (
                source_row.get(
                    "expected_rank_before_catboost",
                    "",
                )
            ),
            "source_expected_rank_after_catboost": (
                source_row.get(
                    "expected_rank_after_catboost",
                    "",
                )
            ),
            "api_success": False,
            "api_error": (
                f"{type(exc).__name__}: {exc}"
            ),
            "http_status": "",
            "actual_decision": "",
            "ranking_mode": "",
            "candidate_count": 0,
            "reranker_applied": False,
            "catboost_shadow_status": "",
            "catboost_shadow_applied": False,
            "catboost_shadow_error": "",
            "api_changed_winner": False,
            "computed_changed_winner": False,
            "winner_change_flag_matches": False,
            "current_winner_company_id": "",
            "current_winner_legal_name": "",
            "shadow_winner_company_id": "",
            "shadow_winner_legal_name": "",
            "current_expected_rank": 0,
            "shadow_expected_rank": 0,
            "rank_change": "API_ERROR",
            "current_top1_correct": False,
            "shadow_top1_correct": False,
            "current_recall_at_k": False,
            "shadow_recall_at_k": False,
            "current_reciprocal_rank": 0.0,
            "shadow_reciprocal_rank": 0.0,
            "current_ndcg": 0.0,
            "shadow_ndcg": 0.0,
            "score_margin": 0.0,
            "catboost_shadow_raw_margin": 0.0,
            "catboost_shadow_normalized_margin": 0.0,
            "current_top5_json": "",
            "shadow_top5_json": "",
            "latency_ms": round(
                latency_ms,
                3,
            ),
        }


def ratio(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator <= 0:
        return None

    return float(
        numerator / denominator
    )


def mean_or_none(
    values: list[float],
) -> float | None:
    if not values:
        return None

    return float(
        statistics.mean(
            values
        )
    )


def summarize_ranked_dataset(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    if not results:
        return {
            "query_count": 0,
            "current_top1_accuracy": None,
            "shadow_top1_accuracy": None,
            "top1_lift": None,
            "current_recall_at_k": None,
            "shadow_recall_at_k": None,
            "current_mrr": None,
            "shadow_mrr": None,
            "mrr_lift": None,
            "current_ndcg": None,
            "shadow_ndcg": None,
            "ndcg_lift": None,
            "improved_count": 0,
            "hurt_count": 0,
            "unchanged_count": 0,
            "current_missing_count": 0,
            "shadow_missing_count": 0,
            "winner_changed_count": 0,
            "current_rank_distribution": {},
            "shadow_rank_distribution": {},
        }

    current_top1_count = sum(
        as_boolean(
            result.get(
                "current_top1_correct"
            )
        )
        for result in results
    )

    shadow_top1_count = sum(
        as_boolean(
            result.get(
                "shadow_top1_correct"
            )
        )
        for result in results
    )

    current_recall_count = sum(
        as_boolean(
            result.get(
                "current_recall_at_k"
            )
        )
        for result in results
    )

    shadow_recall_count = sum(
        as_boolean(
            result.get(
                "shadow_recall_at_k"
            )
        )
        for result in results
    )

    current_top1_accuracy = ratio(
        current_top1_count,
        len(results),
    )

    shadow_top1_accuracy = ratio(
        shadow_top1_count,
        len(results),
    )

    current_mrr = mean_or_none(
        [
            as_float(
                result.get(
                    "current_reciprocal_rank"
                )
            )
            for result in results
        ]
    )

    shadow_mrr = mean_or_none(
        [
            as_float(
                result.get(
                    "shadow_reciprocal_rank"
                )
            )
            for result in results
        ]
    )

    current_ndcg = mean_or_none(
        [
            as_float(
                result.get(
                    "current_ndcg"
                )
            )
            for result in results
        ]
    )

    shadow_ndcg = mean_or_none(
        [
            as_float(
                result.get(
                    "shadow_ndcg"
                )
            )
            for result in results
        ]
    )

    rank_changes = Counter(
        result.get(
            "rank_change",
            "",
        )
        for result in results
    )

    current_rank_distribution = Counter(
        str(
            as_integer(
                result.get(
                    "current_expected_rank"
                )
            )
        )
        for result in results
    )

    shadow_rank_distribution = Counter(
        str(
            as_integer(
                result.get(
                    "shadow_expected_rank"
                )
            )
        )
        for result in results
    )

    winner_changed_count = sum(
        as_boolean(
            result.get(
                "computed_changed_winner"
            )
        )
        for result in results
    )

    return {
        "query_count": len(
            results
        ),
        "current_top1_count": (
            current_top1_count
        ),
        "shadow_top1_count": (
            shadow_top1_count
        ),
        "current_top1_accuracy": (
            current_top1_accuracy
        ),
        "shadow_top1_accuracy": (
            shadow_top1_accuracy
        ),
        "top1_lift": (
            shadow_top1_accuracy
            - current_top1_accuracy
            if (
                shadow_top1_accuracy
                is not None
                and current_top1_accuracy
                is not None
            )
            else None
        ),
        "current_recall_at_k": ratio(
            current_recall_count,
            len(results),
        ),
        "shadow_recall_at_k": ratio(
            shadow_recall_count,
            len(results),
        ),
        "current_mrr": current_mrr,
        "shadow_mrr": shadow_mrr,
        "mrr_lift": (
            shadow_mrr
            - current_mrr
            if (
                shadow_mrr is not None
                and current_mrr is not None
            )
            else None
        ),
        "current_ndcg": current_ndcg,
        "shadow_ndcg": shadow_ndcg,
        "ndcg_lift": (
            shadow_ndcg
            - current_ndcg
            if (
                shadow_ndcg is not None
                and current_ndcg is not None
            )
            else None
        ),
        "improved_count": rank_changes[
            "IMPROVED"
        ],
        "hurt_count": rank_changes[
            "HURT"
        ],
        "unchanged_count": rank_changes[
            "UNCHANGED"
        ],
        "current_missing_count": rank_changes[
            "EXPECTED_NOT_IN_CURRENT_CANDIDATES"
        ],
        "shadow_missing_count": rank_changes[
            "EXPECTED_NOT_IN_SHADOW_RANKING"
        ],
        "winner_changed_count": (
            winner_changed_count
        ),
        "current_rank_distribution": dict(
            sorted(
                current_rank_distribution.items(),
                key=lambda item: int(
                    item[0]
                ),
            )
        ),
        "shadow_rank_distribution": dict(
            sorted(
                shadow_rank_distribution.items(),
                key=lambda item: int(
                    item[0]
                ),
            )
        ),
    }


def summarize_negative_dataset(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    decision_counts = Counter(
        result.get(
            "actual_decision",
            "",
        )
        for result in results
    )

    winner_changed_count = sum(
        as_boolean(
            result.get(
                "computed_changed_winner"
            )
        )
        for result in results
    )

    return {
        "query_count": len(
            results
        ),
        "decision_counts": dict(
            decision_counts
        ),
        "unknown_count": decision_counts[
            "UNKNOWN"
        ],
        "unknown_recall": ratio(
            decision_counts[
                "UNKNOWN"
            ],
            len(results),
        ),
        "manual_review_count": decision_counts[
            "MANUAL_REVIEW"
        ],
        "manual_review_rate": ratio(
            decision_counts[
                "MANUAL_REVIEW"
            ],
            len(results),
        ),
        "false_auto_match_count": decision_counts[
            "AUTO_MATCH"
        ],
        "false_auto_match_rate": ratio(
            decision_counts[
                "AUTO_MATCH"
            ],
            len(results),
        ),
        "winner_changed_count": (
            winner_changed_count
        ),
        "winner_changed_rate": ratio(
            winner_changed_count,
            len(results),
        ),
    }


def calculate_summary(
    results: list[dict[str, Any]],
    runtime_payload: dict[str, Any] | None,
    candidate_limit: int,
    input_counts: dict[str, int],
) -> dict[str, Any]:
    successful_results = [
        result
        for result in results
        if as_boolean(
            result.get(
                "api_success"
            )
        )
    ]

    failed_results = [
        result
        for result in results
        if not as_boolean(
            result.get(
                "api_success"
            )
        )
    ]

    standard_positive_results = [
        result
        for result in successful_results
        if result.get(
            "dataset_type"
        ) == "POSITIVE"
    ]

    hard_positive_results = [
        result
        for result in successful_results
        if result.get(
            "dataset_type"
        ) == "HARD_POSITIVE"
    ]

    negative_results = [
        result
        for result in successful_results
        if result.get(
            "dataset_type"
        ) == "NEGATIVE"
    ]

    combined_ranked_results = (
        standard_positive_results
        + hard_positive_results
    )

    shadow_status_counts = Counter(
        result.get(
            "catboost_shadow_status",
            "",
        )
        for result in successful_results
    )

    applied_count = sum(
        as_boolean(
            result.get(
                "catboost_shadow_applied"
            )
        )
        for result in successful_results
    )

    winner_flag_mismatch_count = sum(
        not as_boolean(
            result.get(
                "winner_change_flag_matches"
            )
        )
        for result in successful_results
    )

    latencies = [
        as_float(
            result.get(
                "latency_ms"
            )
        )
        for result in successful_results
    ]

    runtime_catboost_status: dict[
        str,
        Any,
    ] | None = None

    if runtime_payload:
        possible_status = runtime_payload.get(
            "catboost_shadow"
        )

        if isinstance(
            possible_status,
            dict,
        ):
            runtime_catboost_status = (
                possible_status
            )

    return {
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "candidate_limit": candidate_limit,
        "input_query_counts": input_counts,
        "total_queries": len(
            results
        ),
        "successful_queries": len(
            successful_results
        ),
        "failed_queries": len(
            failed_results
        ),
        "catboost_shadow_applied_count": (
            applied_count
        ),
        "catboost_shadow_applied_rate": ratio(
            applied_count,
            len(successful_results),
        ),
        "catboost_shadow_status_counts": dict(
            shadow_status_counts
        ),
        "winner_change_flag_mismatch_count": (
            winner_flag_mismatch_count
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
        "runtime_catboost_status": (
            runtime_catboost_status
        ),
        "standard_positive": (
            summarize_ranked_dataset(
                standard_positive_results
            )
        ),
        "hard_positive": (
            summarize_ranked_dataset(
                hard_positive_results
            )
        ),
        "combined_positive": (
            summarize_ranked_dataset(
                combined_ranked_results
            )
        ),
        "negative": (
            summarize_negative_dataset(
                negative_results
            )
        ),
        "failed_query_details": [
            {
                "dataset_type": result.get(
                    "dataset_type"
                ),
                "benchmark_id": result.get(
                    "benchmark_id"
                ),
                "query_text": result.get(
                    "query_text"
                ),
                "api_error": result.get(
                    "api_error"
                ),
            }
            for result in failed_results
        ],
    }


def write_results(
    output_path: Path,
    results: list[dict[str, Any]],
) -> None:
    if not results:
        raise ValueError(
            "Yazılacak benchmark sonucu yok."
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames: list[str] = []

    for result in results:
        for field_name in result:
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
            results
        )


def format_percentage(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.2%}"


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    positive_rows = filter_valid_rows(
        read_csv(
            arguments.positive_input,
            "Standart pozitif benchmark",
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

    random_generator = random.Random(
        arguments.seed
    )

    positive_rows = sample_rows(
        positive_rows,
        arguments.positive_sample_size,
        random_generator,
    )

    negative_rows = sample_rows(
        negative_rows,
        arguments.negative_sample_size,
        random_generator,
    )

    hard_positive_rows = sample_rows(
        hard_positive_rows,
        arguments.hard_positive_sample_size,
        random_generator,
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

    input_counts = {
        "POSITIVE": len(
            positive_rows
        ),
        "NEGATIVE": len(
            negative_rows
        ),
        "HARD_POSITIVE": len(
            hard_positive_rows
        ),
    }

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "eft-catboost-shadow-full-benchmark/2.0"
            ),
        }
    )

    runtime_payload: dict[
        str,
        Any,
    ] | None = None

    results: list[
        dict[str, Any]
    ] = []

    try:
        runtime_payload = check_backend(
            session=session,
            backend_url=(
                arguments.backend_url
            ),
            timeout=arguments.timeout,
            expected_model_path=(
                arguments.expected_model_path
            ),
        )

        print(
            "Backend ve matching endpointi hazır."
        )

        if runtime_payload:
            catboost_status = runtime_payload.get(
                "catboost_shadow",
                {},
            )

            print(
                "Aktif CatBoost modeli: "
                f"{catboost_status.get('model_path')}"
            )

            print(
                "Model sürümü: "
                f"{catboost_status.get('model_version')}"
            )

            print(
                "Best iteration: "
                f"{catboost_status.get('best_iteration')}"
            )

        else:
            print(
                "Uyarı: /api/v1/runtime endpointi "
                "bulunamadığı için model yolu "
                "doğrulanamadı."
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
            result = process_query(
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

            results.append(
                result
            )

            if (
                work_index == 1
                or work_index % 25 == 0
                or work_index == total_count
            ):
                successful_count = sum(
                    as_boolean(
                        item.get(
                            "api_success"
                        )
                    )
                    for item in results
                )

                applied_count = sum(
                    as_boolean(
                        item.get(
                            "catboost_shadow_applied"
                        )
                    )
                    for item in results
                )

                improved_count = sum(
                    item.get(
                        "rank_change"
                    ) == "IMPROVED"
                    for item in results
                )

                hurt_count = sum(
                    item.get(
                        "rank_change"
                    ) == "HURT"
                    for item in results
                )

                print(
                    f"{work_index}/{total_count} tamamlandı. "
                    f"Başarılı: {successful_count} | "
                    f"CatBoost applied: {applied_count} | "
                    f"İyileşen: {improved_count} | "
                    f"Kötüleşen: {hurt_count}"
                )

    finally:
        session.close()

    write_results(
        arguments.results_output,
        results,
    )

    summary = calculate_summary(
        results=results,
        runtime_payload=runtime_payload,
        candidate_limit=(
            arguments.candidate_limit
        ),
        input_counts=input_counts,
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

    standard_metrics = summary[
        "standard_positive"
    ]

    hard_metrics = summary[
        "hard_positive"
    ]

    negative_metrics = summary[
        "negative"
    ]

    print()
    print(
        "Genişletilmiş CatBoost shadow benchmark "
        "tamamlandı."
    )

    print(
        "Başarılı sorgu: "
        f"{summary['successful_queries']}/"
        f"{summary['total_queries']}"
    )

    print(
        "Standart pozitif mevcut Top-1: "
        f"{format_percentage(standard_metrics['current_top1_accuracy'])}"
    )

    print(
        "Standart pozitif CatBoost Top-1: "
        f"{format_percentage(standard_metrics['shadow_top1_accuracy'])}"
    )

    print(
        "Hard-positive mevcut Top-1: "
        f"{format_percentage(hard_metrics['current_top1_accuracy'])}"
    )

    print(
        "Hard-positive CatBoost Top-1: "
        f"{format_percentage(hard_metrics['shadow_top1_accuracy'])}"
    )

    print(
        "Hard-positive mevcut MRR: "
        f"{format_percentage(hard_metrics['current_mrr'])}"
    )

    print(
        "Hard-positive CatBoost MRR: "
        f"{format_percentage(hard_metrics['shadow_mrr'])}"
    )

    print(
        "Hard-positive iyileşen: "
        f"{hard_metrics['improved_count']}"
    )

    print(
        "Hard-positive kötüleşen: "
        f"{hard_metrics['hurt_count']}"
    )

    print(
        "Hard-positive değişmeyen: "
        f"{hard_metrics['unchanged_count']}"
    )

    print(
        "Negatif UNKNOWN Recall: "
        f"{format_percentage(negative_metrics['unknown_recall'])}"
    )

    print(
        "Negatif yanlış AUTO_MATCH: "
        f"{negative_metrics['false_auto_match_count']}"
    )

    print(
        "Başarısız sorgu: "
        f"{summary['failed_queries']}"
    )

    print(
        f"Sonuçlar: {arguments.results_output}"
    )

    print(
        f"Özet: {arguments.summary_output}"
    )


if __name__ == "__main__":
    main()