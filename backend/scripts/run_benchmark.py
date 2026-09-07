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
    / "benchmark_queries.csv"
)

DEFAULT_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "benchmark_results.csv"
)

DEFAULT_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "benchmark_summary.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "EFT Company Matcher API üzerinde uçtan uca "
            "benchmark çalıştırır ve beklenen adayın "
            "detaylı skorlarını kaydeder."
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
            "0 bütün sorguları çalıştırır. "
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


def read_csv(
    path: Path,
) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Benchmark dosyası bulunamadı: {path}"
        )

    with path.open(
        mode="r",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        return list(
            csv.DictReader(csv_file)
        )


def percentile(
    values: list[float],
    percentage_value: float,
) -> float | None:
    if not values:
        return None

    sorted_values = sorted(values)

    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (
        percentage_value
        / 100.0
        * (len(sorted_values) - 1)
    )

    lower_index = int(position)

    upper_index = min(
        lower_index + 1,
        len(sorted_values) - 1,
    )

    fraction = position - lower_index

    return (
        sorted_values[lower_index]
        + (
            sorted_values[upper_index]
            - sorted_values[lower_index]
        )
        * fraction
    )


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


def as_optional_float(
    value: Any,
) -> float | None:
    if value in (
        "",
        None,
    ):
        return None

    try:
        return float(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def find_candidate(
    candidates: list[dict[str, Any]],
    company_id: int,
) -> dict[str, Any] | None:
    for candidate in candidates:
        try:
            candidate_company_id = int(
                candidate["company_id"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if candidate_company_id == company_id:
            return candidate

    return None


def build_rank_map(
    candidates: list[dict[str, Any]],
) -> dict[int, int]:
    rank_map: dict[int, int] = {}

    for rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        try:
            company_id = int(
                candidate["company_id"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        rank_map[company_id] = rank

    return rank_map


def sort_by_retrieval_score(
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Reranker uygulanmadan önceki yaklaşık sıralamayı
    candidate alanlarından yeniden oluşturur.

    CompanyIndex sıralamasında unique exact match önce,
    retrieval skoru sonra gelir.
    """

    return sorted(
        candidates,
        key=lambda candidate: (
            bool(
                candidate.get(
                    "unique_exact_match",
                    False,
                )
            ),
            as_float(
                candidate.get(
                    "retrieval_score",
                    candidate.get(
                        "hybrid_score",
                        0.0,
                    ),
                )
            ),
        ),
        reverse=True,
    )


def serialize_top_candidates(
    candidates: list[dict[str, Any]],
    limit: int = 3,
) -> str:
    serialized_candidates: list[
        dict[str, Any]
    ] = []

    for rank, candidate in enumerate(
        candidates[:limit],
        start=1,
    ):
        serialized_candidates.append(
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
                "final_score": candidate.get(
                    "final_score",
                    "",
                ),
                "unique_exact_match": candidate.get(
                    "unique_exact_match",
                    False,
                ),
                "ambiguous_exact_match": candidate.get(
                    "ambiguous_exact_match",
                    False,
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


def determine_reranker_effect(
    expected_rank_before: int | None,
    expected_rank_after: int | None,
) -> str:
    if expected_rank_after is None:
        return "EXPECTED_NOT_FOUND"

    if expected_rank_before is None:
        return "BEFORE_RANK_UNAVAILABLE"

    if expected_rank_after < expected_rank_before:
        return "HELPED"

    if expected_rank_after > expected_rank_before:
        return "HURT"

    return "UNCHANGED"


def calculate_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_count = len(rows)

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

    top1_correct_count = sum(
        as_boolean(
            row.get(
                "top1_correct"
            )
        )
        for row in rows
    )

    top5_correct_count = sum(
        as_boolean(
            row.get(
                "top5_correct"
            )
        )
        for row in rows
    )

    top10_correct_count = sum(
        as_boolean(
            row.get(
                "top10_correct"
            )
        )
        for row in rows
    )

    reciprocal_rank_sum = sum(
        as_float(
            row.get(
                "reciprocal_rank"
            )
        )
        for row in rows
    )

    auto_match_rows = [
        row
        for row in successful_rows
        if row.get(
            "decision"
        ) == "AUTO_MATCH"
    ]

    correct_auto_match_rows = [
        row
        for row in auto_match_rows
        if as_boolean(
            row.get(
                "top1_correct"
            )
        )
    ]

    false_auto_match_rows = [
        row
        for row in auto_match_rows
        if not as_boolean(
            row.get(
                "top1_correct"
            )
        )
    ]

    manual_review_rows = [
        row
        for row in successful_rows
        if row.get(
            "decision"
        ) == "MANUAL_REVIEW"
    ]

    unknown_rows = [
        row
        for row in successful_rows
        if row.get(
            "decision"
        ) == "UNKNOWN"
    ]

    candidate_miss_count = sum(
        1
        for row in successful_rows
        if row.get(
            "true_rank"
        ) in (
            "",
            None,
        )
    )

    latencies = [
        float(
            row["latency_ms"]
        )
        for row in successful_rows
        if row.get(
            "latency_ms"
        ) not in (
            "",
            None,
        )
    ]

    reranker_helped_count = sum(
        row.get(
            "reranker_effect"
        ) == "HELPED"
        for row in successful_rows
    )

    reranker_hurt_count = sum(
        row.get(
            "reranker_effect"
        ) == "HURT"
        for row in successful_rows
    )

    reranker_unchanged_count = sum(
        row.get(
            "reranker_effect"
        ) == "UNCHANGED"
        for row in successful_rows
    )

    reranker_changed_winner_count = sum(
        as_boolean(
            row.get(
                "reranker_changed_winner"
            )
        )
        for row in successful_rows
    )

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
        "top1_accuracy": percentage(
            top1_correct_count,
            total_count,
        ),
        "candidate_recall_at_5": percentage(
            top5_correct_count,
            total_count,
        ),
        "candidate_recall_at_10": percentage(
            top10_correct_count,
            total_count,
        ),
        "mean_reciprocal_rank": (
            round(
                reciprocal_rank_sum
                / total_count,
                6,
            )
            if total_count
            else None
        ),
        "candidate_miss_rate": percentage(
            candidate_miss_count,
            total_count,
        ),
        "auto_match_count": len(
            auto_match_rows
        ),
        "auto_match_coverage": percentage(
            len(auto_match_rows),
            total_count,
        ),
        "auto_match_precision": percentage(
            len(correct_auto_match_rows),
            len(auto_match_rows),
        ),
        "false_auto_match_count": len(
            false_auto_match_rows
        ),
        "false_match_rate": percentage(
            len(false_auto_match_rows),
            total_count,
        ),
        "manual_review_rate": percentage(
            len(manual_review_rows),
            total_count,
        ),
        "unknown_rate": percentage(
            len(unknown_rows),
            total_count,
        ),
        "reranker_helped_count": (
            reranker_helped_count
        ),
        "reranker_hurt_count": (
            reranker_hurt_count
        ),
        "reranker_unchanged_count": (
            reranker_unchanged_count
        ),
        "reranker_changed_winner_count": (
            reranker_changed_winner_count
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


def group_metrics(
    rows: list[dict[str, Any]],
    key: str,
) -> dict[str, Any]:
    groups: dict[
        str,
        list[dict[str, Any]],
    ] = defaultdict(list)

    for row in rows:
        group_name = str(
            row.get(
                key,
                "UNKNOWN",
            )
        )

        groups[
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
            groups.items()
        )
    }


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

    if (
        components.get(
            "embedding_model"
        )
        != "ready"
    ):
        raise RuntimeError(
            "Embedding modeli hazır değil."
        )

    if (
        components.get(
            "reranker_model"
        )
        != "ready"
    ):
        raise RuntimeError(
            "Reranker modeli hazır değil."
        )

    return payload


def run_single_query(
    session: requests.Session,
    backend_url: str,
    row: dict[str, str],
    candidate_limit: int,
    timeout: float,
) -> dict[str, Any]:
    expected_company_id = int(
        row["company_id"]
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

        final_rank_map = build_rank_map(
            candidates
        )

        retrieval_sorted_candidates = (
            sort_by_retrieval_score(
                candidates
            )
        )

        retrieval_rank_map = build_rank_map(
            retrieval_sorted_candidates
        )

        expected_candidate = find_candidate(
            candidates,
            expected_company_id,
        )

        true_rank = final_rank_map.get(
            expected_company_id
        )

        expected_rank_before_rerank = (
            retrieval_rank_map.get(
                expected_company_id
            )
        )

        best_candidate = payload.get(
            "best_candidate"
        )

        predicted_company_id = (
            int(
                best_candidate[
                    "company_id"
                ]
            )
            if best_candidate
            else None
        )

        retrieval_winner = (
            retrieval_sorted_candidates[0]
            if retrieval_sorted_candidates
            else None
        )

        retrieval_winner_company_id = (
            int(
                retrieval_winner[
                    "company_id"
                ]
            )
            if retrieval_winner
            else None
        )

        reranker_changed_winner = (
            retrieval_winner_company_id
            is not None
            and predicted_company_id
            is not None
            and retrieval_winner_company_id
            != predicted_company_id
        )

        expected_retrieval_score = (
            as_optional_float(
                expected_candidate.get(
                    "retrieval_score"
                )
            )
            if expected_candidate
            else None
        )

        expected_reranker_score = (
            as_optional_float(
                expected_candidate.get(
                    "reranker_score"
                )
            )
            if expected_candidate
            else None
        )

        expected_final_score = (
            as_optional_float(
                expected_candidate.get(
                    "final_score"
                )
            )
            if expected_candidate
            else None
        )

        predicted_retrieval_score = (
            as_optional_float(
                best_candidate.get(
                    "retrieval_score"
                )
            )
            if best_candidate
            else None
        )

        predicted_final_score = (
            as_optional_float(
                best_candidate.get(
                    "final_score"
                )
            )
            if best_candidate
            else None
        )

        predicted_expected_final_score_gap = (
            round(
                predicted_final_score
                - expected_final_score,
                6,
            )
            if (
                predicted_final_score
                is not None
                and expected_final_score
                is not None
            )
            else None
        )

        predicted_expected_retrieval_gap = (
            round(
                predicted_retrieval_score
                - expected_retrieval_score,
                6,
            )
            if (
                predicted_retrieval_score
                is not None
                and expected_retrieval_score
                is not None
            )
            else None
        )

        reranker_effect = (
            determine_reranker_effect(
                expected_rank_before=(
                    expected_rank_before_rerank
                ),
                expected_rank_after=(
                    true_rank
                ),
            )
        )

        candidate_ids = [
            int(
                candidate[
                    "company_id"
                ]
            )
            for candidate in candidates
        ]

        return {
            **row,
            "api_success": True,
            "decision": payload.get(
                "decision",
                "",
            ),
            "ranking_mode": payload.get(
                "ranking_mode",
                "",
            ),
            "reranker_applied": payload.get(
                "reranker_applied",
                False,
            ),
            "predicted_company_id": (
                predicted_company_id
                if predicted_company_id
                is not None
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
            "retrieval_winner_company_id": (
                retrieval_winner_company_id
                if retrieval_winner_company_id
                is not None
                else ""
            ),
            "retrieval_winner_legal_name": (
                retrieval_winner.get(
                    "legal_name",
                    "",
                )
                if retrieval_winner
                else ""
            ),
            "predicted_retrieval_score": (
                predicted_retrieval_score
                if predicted_retrieval_score
                is not None
                else ""
            ),
            "predicted_reranker_score": (
                best_candidate.get(
                    "reranker_score",
                    "",
                )
                if best_candidate
                else ""
            ),
            "predicted_final_score": (
                predicted_final_score
                if predicted_final_score
                is not None
                else ""
            ),
            "expected_candidate_found": (
                expected_candidate
                is not None
            ),
            "expected_matched_alias": (
                expected_candidate.get(
                    "matched_alias",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_alias_type": (
                expected_candidate.get(
                    "alias_type",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_retrieval_score": (
                expected_retrieval_score
                if expected_retrieval_score
                is not None
                else ""
            ),
            "expected_reranker_score": (
                expected_reranker_score
                if expected_reranker_score
                is not None
                else ""
            ),
            "expected_final_score": (
                expected_final_score
                if expected_final_score
                is not None
                else ""
            ),
            "expected_embedding_score": (
                expected_candidate.get(
                    "embedding_score",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_char_tfidf_score": (
                expected_candidate.get(
                    "char_tfidf_score",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_fuzzy_wratio_score": (
                expected_candidate.get(
                    "fuzzy_wratio_score",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_identifier_score": (
                expected_candidate.get(
                    "identifier_score",
                    "",
                )
                if expected_candidate
                else ""
            ),
            "expected_rank_before_rerank": (
                expected_rank_before_rerank
                if expected_rank_before_rerank
                is not None
                else ""
            ),
            "expected_rank_after_rerank": (
                true_rank
                if true_rank is not None
                else ""
            ),
            "predicted_expected_retrieval_gap": (
                predicted_expected_retrieval_gap
                if predicted_expected_retrieval_gap
                is not None
                else ""
            ),
            "predicted_expected_final_score_gap": (
                predicted_expected_final_score_gap
                if predicted_expected_final_score_gap
                is not None
                else ""
            ),
            "reranker_effect": (
                reranker_effect
            ),
            "reranker_changed_winner": (
                reranker_changed_winner
            ),
            "score_margin": payload.get(
                "score_margin",
                "",
            ),
            "true_rank": (
                true_rank
                if true_rank is not None
                else ""
            ),
            "top1_correct": (
                true_rank == 1
            ),
            "top5_correct": (
                true_rank is not None
                and true_rank <= 5
            ),
            "top10_correct": (
                true_rank is not None
                and true_rank <= 10
            ),
            "reciprocal_rank": (
                round(
                    1.0 / true_rank,
                    6,
                )
                if true_rank is not None
                else 0.0
            ),
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "candidate_company_ids": (
                "|".join(
                    str(company_id)
                    for company_id
                    in candidate_ids
                )
            ),
            "top3_candidates_json": (
                serialize_top_candidates(
                    candidates,
                    limit=3,
                )
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
            "decision": "",
            "ranking_mode": "",
            "reranker_applied": False,
            "predicted_company_id": "",
            "predicted_legal_name": "",
            "predicted_alias": "",
            "retrieval_winner_company_id": "",
            "retrieval_winner_legal_name": "",
            "predicted_retrieval_score": "",
            "predicted_reranker_score": "",
            "predicted_final_score": "",
            "expected_candidate_found": False,
            "expected_matched_alias": "",
            "expected_alias_type": "",
            "expected_retrieval_score": "",
            "expected_reranker_score": "",
            "expected_final_score": "",
            "expected_embedding_score": "",
            "expected_char_tfidf_score": "",
            "expected_fuzzy_wratio_score": "",
            "expected_identifier_score": "",
            "expected_rank_before_rerank": "",
            "expected_rank_after_rerank": "",
            "predicted_expected_retrieval_gap": "",
            "predicted_expected_final_score_gap": "",
            "reranker_effect": "",
            "reranker_changed_winner": False,
            "score_margin": "",
            "true_rank": "",
            "top1_correct": False,
            "top5_correct": False,
            "top10_correct": False,
            "reciprocal_rank": 0.0,
            "latency_ms": round(
                latency_ms,
                3,
            ),
            "candidate_company_ids": "",
            "top3_candidates_json": "",
            "error": (
                f"{type(exc).__name__}: {exc}"
            ),
        }


def main() -> None:
    arguments = parse_arguments()

    if arguments.candidate_limit < 1:
        raise ValueError(
            "--candidate-limit en az 1 olmalıdır."
        )

    benchmark_rows = read_csv(
        arguments.input
    )

    if not benchmark_rows:
        raise ValueError(
            "Benchmark veri seti boş."
        )

    if (
        arguments.sample_size > 0
        and arguments.sample_size
        < len(benchmark_rows)
    ):
        rng = random.Random(
            arguments.seed
        )

        benchmark_rows = rng.sample(
            benchmark_rows,
            arguments.sample_size,
        )

    session = requests.Session()

    session.headers.update(
        {
            "Accept": "application/json",
            "User-Agent": (
                "eft-company-benchmark/1.1"
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
            benchmark_rows
        )

        for index, row in enumerate(
            benchmark_rows,
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
                index == 1
                or index % 25 == 0
                or index == total_count
            ):
                top1_count = sum(
                    as_boolean(
                        result_row.get(
                            "top1_correct"
                        )
                    )
                    for result_row
                    in results
                )

                print(
                    f"{index}/{total_count} "
                    "tamamlandı. "
                    "Geçici Top-1: "
                    f"{top1_count / index:.2%}"
                )

    finally:
        session.close()

    if not results:
        raise RuntimeError(
            "Benchmark sonucu üretilemedi."
        )

    arguments.results.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result_fieldnames = list(
        results[0].keys()
    )

    with arguments.results.open(
        mode="w",
        encoding="utf-8-sig",
        newline="",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=result_fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        writer.writerows(
            results
        )

    strict_results = [
        row
        for row in results
        if as_boolean(
            row.get(
                "strict_evaluation_eligible",
                True,
            )
        )
    ]

    collision_results = [
        row
        for row in results
        if as_boolean(
            row.get(
                "benchmark_collision",
                False,
            )
        )
    ]

    ambiguous_results = [
        row
        for row in results
        if as_boolean(
            row.get(
                "ambiguous_query",
                False,
            )
        )
    ]

    overall_metrics = (
        calculate_metrics(
            results
        )
    )

    strict_overall_metrics = (
        calculate_metrics(
            strict_results
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
        "strict_sample_size": len(
            strict_results
        ),
        "collision_query_count": len(
            collision_results
        ),
        "ambiguous_query_count": len(
            ambiguous_results
        ),
        "overall": overall_metrics,
        "strict_overall": (
            strict_overall_metrics
        ),
        "by_difficulty": group_metrics(
            results,
            "difficulty",
        ),
        "strict_by_difficulty": (
            group_metrics(
                strict_results,
                "difficulty",
            )
        ),
        "by_corruption_type": (
            group_metrics(
                results,
                "corruption_type",
            )
        ),
        "by_reranker_effect": (
            group_metrics(
                results,
                "reranker_effect",
            )
        ),
        "backend_health": health,
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

    print()
    print("Benchmark tamamlandı.")

    print(
        "Tüm kayıtlar Top-1: "
        f"{overall_metrics['top1_accuracy']:.2%}"
    )

    if strict_results:
        print(
            "Strict Top-1 Accuracy: "
            f"{strict_overall_metrics['top1_accuracy']:.2%}"
        )

        print(
            "Strict Recall@10: "
            f"{strict_overall_metrics['candidate_recall_at_10']:.2%}"
        )

        strict_auto_precision = (
            strict_overall_metrics[
                "auto_match_precision"
            ]
        )

        print(
            "Strict AUTO_MATCH Precision: "
            + (
                f"{strict_auto_precision:.2%}"
                if strict_auto_precision
                is not None
                else "hesaplanamadı"
            )
        )

    print(
        "Reranker doğru adayı yükseltti: "
        f"{overall_metrics['reranker_helped_count']}"
    )

    print(
        "Reranker doğru adayı düşürdü: "
        f"{overall_metrics['reranker_hurt_count']}"
    )

    print(
        f"Sonuçlar: {arguments.results}"
    )

    print(
        f"Özet: {arguments.summary}"
    )


if __name__ == "__main__":
    main()