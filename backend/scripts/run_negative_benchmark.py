from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "negative_benchmark_queries.csv"
)

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "negative_benchmark_results.csv"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "negative_benchmark_summary.json"
)


DECISION_RISK_PRIORITY = {
    "AUTO_MATCH": 0,
    "MANUAL_REVIEW": 1,
    "UNKNOWN": 2,
    "": 3,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Negatif EFT açıklamalarını Company Matcher API "
            "üzerinde çalıştırır ve UNKNOWN performansını ölçer."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
    )

    parser.add_argument(
        "--results",
        type=Path,
        default=DEFAULT_RESULTS_PATH,
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
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
        "--sample-size",
        type=int,
        default=0,
        help=(
            "0 bütün negatif sorguları çalıştırır. "
            "Pozitif değer rastgele örnek alır."
        ),
    )

    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
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
    if arguments.candidate_limit < 1:
        raise ValueError(
            "--candidate-limit en az 1 olmalıdır."
        )

    if arguments.sample_size < 0:
        raise ValueError(
            "--sample-size negatif olamaz."
        )

    if arguments.timeout <= 0:
        raise ValueError(
            "--timeout pozitif olmalıdır."
        )


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Negatif benchmark dosyası bulunamadı: {path}"
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
            "Negatif benchmark veri seti boş."
        )

    required_columns = {
        "benchmark_id",
        "negative_category",
        "query_text",
        "expected_decision",
    }

    missing_columns = required_columns.difference(
        rows[0].keys()
    )

    if missing_columns:
        raise ValueError(
            "Negatif benchmark CSV eksik kolon içeriyor: "
            f"{sorted(missing_columns)}"
        )

    return rows


def as_boolean(
    value: Any,
) -> bool:
    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def as_float(
    value: Any,
    default: float = 0.0,
) -> float:
    if value in (
        "",
        None,
    ):
        return default

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return default


def percentage(
    numerator: int,
    denominator: int,
) -> float | None:
    if denominator == 0:
        return None

    return round(
        numerator / denominator,
        6,
    )


def percentile(
    values: list[float],
    percentile_value: float,
) -> float | None:
    if not values:
        return None

    sorted_values = sorted(
        values
    )

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        percentile_value
        / 100.0
        * (len(sorted_values) - 1)
    )

    lower_index = int(
        position
    )

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = (
        position
        - lower_index
    )

    return (
        sorted_values[lower_index]
        + (
            sorted_values[upper_index]
            - sorted_values[lower_index]
        )
        * fraction
    )


def check_backend(
    session: requests.Session,
    backend_url: str,
    timeout: float,
) -> dict[str, Any]:
    endpoint = (
        f"{backend_url.rstrip('/')}"
        "/api/v1/health"
    )

    response = session.get(
        endpoint,
        timeout=timeout,
    )

    response.raise_for_status()

    payload = response.json()

    components = payload.get(
        "components",
        {},
    )

    if components.get(
        "company_index"
    ) != "ready":
        raise RuntimeError(
            "Company index hazır değil."
        )

    if components.get(
        "embedding_model"
    ) != "ready":
        raise RuntimeError(
            "Embedding modeli hazır değil."
        )

    if components.get(
        "reranker_model"
    ) != "ready":
        raise RuntimeError(
            "Reranker modeli hazır değil."
        )

    return payload


def serialize_top_candidates(
    candidates: list[dict[str, Any]],
    limit: int = 3,
) -> str:
    output: list[dict[str, Any]] = []

    for rank, candidate in enumerate(
        candidates[:limit],
        start=1,
    ):
        output.append(
            {
                "rank": rank,
                "company_id": candidate.get(
                    "company_id"
                ),
                "legal_name": candidate.get(
                    "legal_name",
                    "",
                ),
                "matched_alias": candidate.get(
                    "matched_alias",
                    "",
                ),
                "retrieval_score": candidate.get(
                    "retrieval_score",
                    "",
                ),
                "reranker_score": candidate.get(
                    "reranker_score",
                    "",
                ),
                "adjusted_reranker_score": candidate.get(
                    "adjusted_reranker_score",
                    "",
                ),
                "lexical_evidence": candidate.get(
                    "lexical_evidence",
                    "",
                ),
                "final_score": candidate.get(
                    "final_score",
                    "",
                ),
            }
        )

    return json.dumps(
        output,
        ensure_ascii=False,
        separators=(
            ",",
            ":",
        ),
    )


def classify_negative_result(
    *,
    decision: str,
    best_candidate: dict[str, Any] | None,
) -> str:
    if decision == "UNKNOWN":
        return "CORRECT_UNKNOWN"

    if decision == "AUTO_MATCH":
        return "CRITICAL_FALSE_AUTO_MATCH"

    if decision == "MANUAL_REVIEW":
        return "FALSE_MANUAL_REVIEW"

    if best_candidate is not None:
        return "UNEXPECTED_CANDIDATE_RESULT"

    return "UNCLASSIFIED"


def run_single_query(
    *,
    session: requests.Session,
    backend_url: str,
    row: dict[str, str],
    candidate_limit: int,
    timeout: float,
) -> dict[str, Any]:
    endpoint = (
        f"{backend_url.rstrip('/')}"
        "/api/v1/matching/candidates"
    )

    started_at = time.perf_counter()

    try:
        response = session.post(
            endpoint,
            json={
                "text": row["query_text"],
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
            raise ValueError(
                "API candidates alanı liste değil."
            )

        best_candidate = payload.get(
            "best_candidate"
        )

        decision = str(
            payload.get(
                "decision",
                "",
            )
        )

        expected_decision = str(
            row.get(
                "expected_decision",
                "UNKNOWN",
            )
        )

        correct_unknown = (
            decision == "UNKNOWN"
        )

        false_auto_match = (
            decision == "AUTO_MATCH"
        )

        false_manual_review = (
            decision == "MANUAL_REVIEW"
        )

        false_match = (
            decision != expected_decision
        )

        best_final_score = (
            as_float(
                best_candidate.get(
                    "final_score"
                )
            )
            if best_candidate
            else 0.0
        )

        best_retrieval_score = (
            as_float(
                best_candidate.get(
                    "retrieval_score"
                )
            )
            if best_candidate
            else 0.0
        )

        best_reranker_score = (
            as_float(
                best_candidate.get(
                    "reranker_score"
                )
            )
            if best_candidate
            else 0.0
        )

        best_adjusted_reranker_score = (
            as_float(
                best_candidate.get(
                    "adjusted_reranker_score"
                )
            )
            if best_candidate
            else 0.0
        )

        best_lexical_evidence = (
            as_float(
                best_candidate.get(
                    "lexical_evidence"
                )
            )
            if best_candidate
            else 0.0
        )

        risk_type = (
            classify_negative_result(
                decision=decision,
                best_candidate=best_candidate,
            )
        )

        return {
            **row,
            "api_success": True,
            "actual_decision": decision,
            "correct_unknown": (
                correct_unknown
            ),
            "false_match": (
                false_match
            ),
            "false_auto_match": (
                false_auto_match
            ),
            "false_manual_review": (
                false_manual_review
            ),
            "risk_type": (
                risk_type
            ),
            "predicted_company_id": (
                best_candidate.get(
                    "company_id",
                    "",
                )
                if best_candidate
                else ""
            ),
            "predicted_legal_name": (
                best_candidate.get(
                    "legal_name",
                    "",
                )
                if best_candidate
                else ""
            ),
            "predicted_alias": (
                best_candidate.get(
                    "matched_alias",
                    "",
                )
                if best_candidate
                else ""
            ),
            "best_retrieval_score": round(
                best_retrieval_score,
                6,
            ),
            "best_reranker_score": round(
                best_reranker_score,
                6,
            ),
            "best_adjusted_reranker_score": round(
                best_adjusted_reranker_score,
                6,
            ),
            "best_lexical_evidence": round(
                best_lexical_evidence,
                6,
            ),
            "best_final_score": round(
                best_final_score,
                6,
            ),
            "score_margin": payload.get(
                "score_margin",
                0.0,
            ),
            "ranking_mode": payload.get(
                "ranking_mode",
                "",
            ),
            "reranker_applied": payload.get(
                "reranker_applied",
                False,
            ),
            "query_length": payload.get(
                "query_length",
                0,
            ),
            "query_token_count": payload.get(
                "query_token_count",
                0,
            ),
            "reranker_gap": payload.get(
                "reranker_gap",
                0.0,
            ),
            "base_reranker_weight": payload.get(
                "base_reranker_weight",
                0.0,
            ),
            "candidate_count": len(
                candidates
            ),
            "top3_candidates_json": (
                serialize_top_candidates(
                    candidates,
                    limit=3,
                )
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "error": "",
        }

    except Exception as exc:
        latency_ms = (
            time.perf_counter()
            - started_at
        ) * 1000.0

        return {
            **row,
            "api_success": False,
            "actual_decision": "",
            "correct_unknown": False,
            "false_match": False,
            "false_auto_match": False,
            "false_manual_review": False,
            "risk_type": "API_ERROR",
            "predicted_company_id": "",
            "predicted_legal_name": "",
            "predicted_alias": "",
            "best_retrieval_score": "",
            "best_reranker_score": "",
            "best_adjusted_reranker_score": "",
            "best_lexical_evidence": "",
            "best_final_score": "",
            "score_margin": "",
            "ranking_mode": "",
            "reranker_applied": False,
            "query_length": "",
            "query_token_count": "",
            "reranker_gap": "",
            "base_reranker_weight": "",
            "candidate_count": 0,
            "top3_candidates_json": "",
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_count = len(
        rows
    )

    successful_rows = [
        row
        for row in rows
        if as_boolean(
            row.get(
                "api_success"
            )
        )
    ]

    successful_count = len(
        successful_rows
    )

    unknown_count = sum(
        row.get(
            "actual_decision"
        ) == "UNKNOWN"
        for row in successful_rows
    )

    auto_match_count = sum(
        row.get(
            "actual_decision"
        ) == "AUTO_MATCH"
        for row in successful_rows
    )

    manual_review_count = sum(
        row.get(
            "actual_decision"
        ) == "MANUAL_REVIEW"
        for row in successful_rows
    )

    false_match_count = sum(
        row.get(
            "actual_decision"
        ) != "UNKNOWN"
        for row in successful_rows
    )

    candidate_exposed_count = sum(
        int(
            row.get(
                "candidate_count",
                0,
            )
        )
        > 0
        for row in successful_rows
    )

    latencies = [
        as_float(
            row.get(
                "latency_ms"
            )
        )
        for row in successful_rows
    ]

    final_scores = [
        as_float(
            row.get(
                "best_final_score"
            )
        )
        for row in successful_rows
        if row.get(
            "predicted_company_id"
        ) not in (
            "",
            None,
        )
    ]

    false_match_scores = [
        as_float(
            row.get(
                "best_final_score"
            )
        )
        for row in successful_rows
        if row.get(
            "actual_decision"
        ) != "UNKNOWN"
    ]

    return {
        "total_queries": total_count,
        "successful_queries": successful_count,
        "failed_queries": (
            total_count
            - successful_count
        ),
        "request_success_rate": percentage(
            successful_count,
            total_count,
        ),
        "unknown_count": (
            unknown_count
        ),
        "unknown_recall": percentage(
            unknown_count,
            successful_count,
        ),
        "false_match_count": (
            false_match_count
        ),
        "negative_false_match_rate": percentage(
            false_match_count,
            successful_count,
        ),
        "false_auto_match_count": (
            auto_match_count
        ),
        "negative_auto_match_rate": percentage(
            auto_match_count,
            successful_count,
        ),
        "false_manual_review_count": (
            manual_review_count
        ),
        "negative_manual_review_rate": percentage(
            manual_review_count,
            successful_count,
        ),
        "candidate_exposed_count": (
            candidate_exposed_count
        ),
        "candidate_exposure_rate": percentage(
            candidate_exposed_count,
            successful_count,
        ),
        "average_best_final_score": (
            round(
                statistics.mean(
                    final_scores
                ),
                6,
            )
            if final_scores
            else None
        ),
        "average_false_match_score": (
            round(
                statistics.mean(
                    false_match_scores
                ),
                6,
            )
            if false_match_scores
            else None
        ),
        "maximum_false_match_score": (
            round(
                max(
                    false_match_scores
                ),
                6,
            )
            if false_match_scores
            else None
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
        "p95_latency_ms": (
            round(
                percentile(
                    latencies,
                    95,
                )
                or 0.0,
                3,
            )
            if latencies
            else None
        ),
    }


def calculate_group_metrics(
    rows: list[dict[str, Any]],
    group_column: str,
) -> dict[str, Any]:
    grouped_rows: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        group_name = str(
            row.get(
                group_column,
                "UNKNOWN",
            )
        )

        grouped_rows[
            group_name
        ].append(
            row
        )

    return {
        group_name: calculate_metrics(
            group_rows
        )
        for group_name, group_rows
        in sorted(
            grouped_rows.items()
        )
    }


def build_riskiest_queries(
    rows: list[dict[str, Any]],
    limit: int = 30,
) -> list[dict[str, Any]]:
    successful_rows = [
        row
        for row in rows
        if as_boolean(
            row.get(
                "api_success"
            )
        )
    ]

    sorted_rows = sorted(
        successful_rows,
        key=lambda row: (
            DECISION_RISK_PRIORITY.get(
                str(
                    row.get(
                        "actual_decision",
                        "",
                    )
                ),
                99,
            ),
            -as_float(
                row.get(
                    "best_final_score"
                )
            ),
            -as_float(
                row.get(
                    "score_margin"
                )
            ),
        ),
    )

    output: list[
        dict[str, Any]
    ] = []

    for row in sorted_rows[
        :limit
    ]:
        output.append(
            {
                "benchmark_id": row.get(
                    "benchmark_id"
                ),
                "negative_category": row.get(
                    "negative_category"
                ),
                "query_text": row.get(
                    "query_text"
                ),
                "actual_decision": row.get(
                    "actual_decision"
                ),
                "predicted_company_id": row.get(
                    "predicted_company_id"
                ),
                "predicted_legal_name": row.get(
                    "predicted_legal_name"
                ),
                "predicted_alias": row.get(
                    "predicted_alias"
                ),
                "best_final_score": row.get(
                    "best_final_score"
                ),
                "best_retrieval_score": row.get(
                    "best_retrieval_score"
                ),
                "best_reranker_score": row.get(
                    "best_reranker_score"
                ),
                "best_adjusted_reranker_score": row.get(
                    "best_adjusted_reranker_score"
                ),
                "best_lexical_evidence": row.get(
                    "best_lexical_evidence"
                ),
                "score_margin": row.get(
                    "score_margin"
                ),
                "closest_known_alias": row.get(
                    "closest_known_alias"
                ),
                "closest_known_similarity": row.get(
                    "closest_known_similarity"
                ),
            }
        )

    return output


def write_results(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        raise ValueError(
            "Yazılacak negatif benchmark sonucu yok."
        )

    fieldnames: list[str] = []

    seen_fields: set[str] = set()

    for row in rows:
        for field_name in row:
            if field_name in seen_fields:
                continue

            seen_fields.add(
                field_name
            )

            fieldnames.append(
                field_name
            )

    with path.open(
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
            rows
        )


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    benchmark_rows = read_csv(
        arguments.input
    )

    valid_rows = [
        row
        for row in benchmark_rows
        if as_boolean(
            row.get(
                "benchmark_valid",
                True,
            )
        )
    ]

    if not valid_rows:
        raise ValueError(
            "Geçerli negatif benchmark kaydı bulunamadı."
        )

    if (
        arguments.sample_size > 0
        and arguments.sample_size
        < len(valid_rows)
    ):
        rng = random.Random(
            arguments.seed
        )

        valid_rows = rng.sample(
            valid_rows,
            arguments.sample_size,
        )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "eft-company-negative-benchmark/1.0"
            ),
        }
    )

    results: list[
        dict[str, Any]
    ] = []

    health: dict[str, Any] = {}

    try:
        health = check_backend(
            session=session,
            backend_url=(
                arguments.backend_url
            ),
            timeout=arguments.timeout,
        )

        components = health.get(
            "components",
            {},
        )

        index_stats = components.get(
            "company_index_stats",
            {},
        )

        print("Backend hazır.")

        print(
            "Şirket sayısı: "
            f"{index_stats.get('company_count')}"
        )

        print(
            "Alias sayısı: "
            f"{index_stats.get('alias_count')}"
        )

        total_count = len(
            valid_rows
        )

        for row_index, row in enumerate(
            valid_rows,
            start=1,
        ):
            result = run_single_query(
                session=session,
                backend_url=(
                    arguments.backend_url
                ),
                row=row,
                candidate_limit=(
                    arguments.candidate_limit
                ),
                timeout=arguments.timeout,
            )

            results.append(
                result
            )

            if (
                row_index == 1
                or row_index % 25 == 0
                or row_index == total_count
            ):
                completed_metrics = (
                    calculate_metrics(
                        results
                    )
                )

                unknown_recall = (
                    completed_metrics[
                        "unknown_recall"
                    ]
                )

                unknown_text = (
                    f"{unknown_recall:.2%}"
                    if unknown_recall
                    is not None
                    else "N/A"
                )

                print(
                    f"{row_index}/{total_count} tamamlandı. "
                    f"Geçici UNKNOWN Recall: {unknown_text} | "
                    "Yanlış AUTO_MATCH: "
                    f"{completed_metrics['false_auto_match_count']}"
                )

    finally:
        session.close()

    write_results(
        path=arguments.results,
        rows=results,
    )

    overall_metrics = (
        calculate_metrics(
            results
        )
    )

    summary = {
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "backend_url": (
            arguments.backend_url
        ),
        "candidate_limit": (
            arguments.candidate_limit
        ),
        "sample_size": len(
            results
        ),
        "overall": (
            overall_metrics
        ),
        "by_negative_category": (
            calculate_group_metrics(
                results,
                "negative_category",
            )
        ),
        "by_actual_decision": (
            calculate_group_metrics(
                results,
                "actual_decision",
            )
        ),
        "riskiest_queries": (
            build_riskiest_queries(
                results,
                limit=30,
            )
        ),
        "backend_health": (
            health
        ),
    }

    arguments.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.summary.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    unknown_recall = (
        overall_metrics[
            "unknown_recall"
        ]
    )

    false_match_rate = (
        overall_metrics[
            "negative_false_match_rate"
        ]
    )

    auto_match_rate = (
        overall_metrics[
            "negative_auto_match_rate"
        ]
    )

    print()
    print(
        "Negatif benchmark tamamlandı."
    )

    print(
        "UNKNOWN Recall: "
        + (
            f"{unknown_recall:.2%}"
            if unknown_recall is not None
            else "N/A"
        )
    )

    print(
        "Negatif false-match rate: "
        + (
            f"{false_match_rate:.2%}"
            if false_match_rate is not None
            else "N/A"
        )
    )

    print(
        "Negatif AUTO_MATCH rate: "
        + (
            f"{auto_match_rate:.2%}"
            if auto_match_rate is not None
            else "N/A"
        )
    )

    print(
        "Yanlış AUTO_MATCH sayısı: "
        f"{overall_metrics['false_auto_match_count']}"
    )

    print(
        "Yanlış MANUAL_REVIEW sayısı: "
        f"{overall_metrics['false_manual_review_count']}"
    )

    print(
        f"Sonuçlar: {arguments.results}"
    )

    print(
        f"Özet: {arguments.summary}"
    )


if __name__ == "__main__":
    main()