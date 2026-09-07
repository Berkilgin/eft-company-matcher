from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import (
    CatBoostError,
    CatBoostRanker,
    Pool,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_INPUT = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "candidate_feature_dataset.csv"
)

DEFAULT_MODEL_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "catboost"
    / "company_ranker.cbm"
)

DEFAULT_METADATA_OUTPUT = (
    PROJECT_ROOT
    / "models"
    / "catboost"
    / "company_ranker_metadata.json"
)


# CatBoost shadow çıktıları, etiketler, metinler ve kimlikler
# feature listesine özellikle dahil edilmez.
FEATURE_COLUMNS: tuple[str, ...] = (
    "exact_match",
    "unique_exact_match",
    "ambiguous_exact_match",
    "identifier_score",
    "identifier_overlap",
    "char_wb_tfidf_score",
    "raw_char_tfidf_score",
    "char_tfidf_score",
    "jaro_winkler_score",
    "token_prefix_score",
    "fuzzy_ratio_score",
    "fuzzy_partial_ratio_score",
    "fuzzy_token_set_score",
    "fuzzy_wratio_score",
    "embedding_score",
    "hybrid_score",
    "retrieval_score",
    "retrieval_rank",
    "reranker_score",
    "adjusted_reranker_score",
    "lexical_evidence",
    "retrieval_weight",
    "reranker_weight",
    "reranker_gap",
    "candidate_reranker_gap",
    "base_reranker_weight",
    "query_length",
    "query_token_count",
    "final_score",
    "reranked",
)


REQUIRED_COLUMNS: tuple[str, ...] = (
    "dataset_type",
    "group_id",
    "expected_company_id",
    "company_id",
    "label",
    "ranker_eligible",
    "candidate_position",
    "current_final_rank",
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standart pozitif ve hard-positive sorgulardan "
            "CatBoostRanker modeli eğitir."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )

    parser.add_argument(
        "--model-output",
        type=Path,
        default=DEFAULT_MODEL_OUTPUT,
    )

    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_OUTPUT,
    )

    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=800,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
    )

    parser.add_argument(
        "--depth",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--l2-leaf-reg",
        type=float,
        default=5.0,
    )

    parser.add_argument(
        "--early-stopping-rounds",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--metric-period",
        type=int,
        default=25,
    )

    parser.add_argument(
        "--standard-positive-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--hard-positive-weight",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--hard-rank-weight-step",
        type=float,
        default=0.15,
        help=(
            "Hard-positive beklenen sırası yükseldikçe "
            "grup ağırlığına eklenecek değer."
        ),
    )

    parser.add_argument(
        "--task-type",
        choices=(
            "AUTO",
            "CPU",
            "GPU",
        ),
        default="AUTO",
    )

    parser.add_argument(
        "--devices",
        type=str,
        default="0",
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
    if not 0.05 <= arguments.validation_ratio <= 0.50:
        raise ValueError(
            "--validation-ratio 0.05 ile 0.50 arasında olmalıdır."
        )

    if arguments.iterations < 10:
        raise ValueError(
            "--iterations en az 10 olmalıdır."
        )

    if arguments.learning_rate <= 0:
        raise ValueError(
            "--learning-rate pozitif olmalıdır."
        )

    if arguments.depth < 2:
        raise ValueError(
            "--depth en az 2 olmalıdır."
        )

    if arguments.l2_leaf_reg < 0:
        raise ValueError(
            "--l2-leaf-reg negatif olamaz."
        )

    if arguments.early_stopping_rounds < 1:
        raise ValueError(
            "--early-stopping-rounds en az 1 olmalıdır."
        )

    if arguments.metric_period < 1:
        raise ValueError(
            "--metric-period en az 1 olmalıdır."
        )

    if arguments.standard_positive_weight <= 0:
        raise ValueError(
            "--standard-positive-weight pozitif olmalıdır."
        )

    if arguments.hard_positive_weight <= 0:
        raise ValueError(
            "--hard-positive-weight pozitif olmalıdır."
        )

    if arguments.hard_rank_weight_step < 0:
        raise ValueError(
            "--hard-rank-weight-step negatif olamaz."
        )


def as_integer_series(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .fillna(0)
        .astype(np.int64)
    )


def as_float_series(
    series: pd.Series,
) -> pd.Series:
    return (
        pd.to_numeric(
            series,
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(0.0)
        .astype(np.float64)
    )


def normalize_identifier_series(
    series: pd.Series,
) -> pd.Series:
    normalized = (
        series
        .fillna("")
        .astype(str)
        .str.strip()
    )

    return normalized.str.replace(
        r"\.0$",
        "",
        regex=True,
    )


def read_dataset(
    path: Path,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Candidate feature veri seti bulunamadı: {path}"
        )

    dataframe = pd.read_csv(
        path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    if dataframe.empty:
        raise ValueError(
            "Candidate feature veri seti boş."
        )

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Candidate feature veri setinde gerekli kolonlar "
            f"eksik: {missing_columns}"
        )

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "CatBoost feature kolonları eksik: "
            f"{missing_features}"
        )

    dataframe["dataset_type"] = (
        dataframe["dataset_type"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    dataframe["group_id"] = (
        dataframe["group_id"]
        .astype(str)
        .str.strip()
    )

    dataframe["expected_company_id"] = (
        normalize_identifier_series(
            dataframe["expected_company_id"]
        )
    )

    dataframe["company_id"] = (
        normalize_identifier_series(
            dataframe["company_id"]
        )
    )

    dataframe["label"] = as_integer_series(
        dataframe["label"]
    )

    dataframe["ranker_eligible"] = as_integer_series(
        dataframe["ranker_eligible"]
    )

    dataframe["candidate_position"] = as_integer_series(
        dataframe["candidate_position"]
    )

    dataframe["current_final_rank"] = as_integer_series(
        dataframe["current_final_rank"]
    )

    if "expected_rank" in dataframe.columns:
        dataframe["expected_rank"] = as_integer_series(
            dataframe["expected_rank"]
        )

    else:
        dataframe["expected_rank"] = 0

    for feature_column in FEATURE_COLUMNS:
        dataframe[feature_column] = as_float_series(
            dataframe[feature_column]
        )

    return dataframe


def filter_eligible_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    eligible = dataframe.loc[
        dataframe["ranker_eligible"] == 1
    ].copy()

    eligible = eligible.loc[
        eligible["dataset_type"].isin(
            [
                "POSITIVE",
                "HARD_POSITIVE",
            ]
        )
    ].copy()

    if eligible.empty:
        raise ValueError(
            "Ranker eğitimine uygun pozitif grup bulunamadı."
        )

    return eligible


def validate_groups(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    invalid_groups: list[dict[str, Any]] = []

    group_dataset_types: Counter[str] = Counter()

    group_company_ids: set[str] = set()

    for group_id, group in dataframe.groupby(
        "group_id",
        sort=False,
    ):
        candidate_count = len(group)

        positive_count = int(
            group["label"].sum()
        )

        dataset_types = set(
            group["dataset_type"]
        )

        expected_company_ids = {
            value
            for value in group[
                "expected_company_id"
            ]
            if value
        }

        valid = (
            candidate_count >= 2
            and positive_count == 1
            and len(dataset_types) == 1
            and len(expected_company_ids) == 1
        )

        if not valid:
            invalid_groups.append(
                {
                    "group_id": group_id,
                    "candidate_count": candidate_count,
                    "positive_count": positive_count,
                    "dataset_types": sorted(
                        dataset_types
                    ),
                    "expected_company_ids": sorted(
                        expected_company_ids
                    ),
                }
            )

            continue

        dataset_type = next(
            iter(dataset_types)
        )

        expected_company_id = next(
            iter(expected_company_ids)
        )

        group_dataset_types[
            dataset_type
        ] += 1

        group_company_ids.add(
            expected_company_id
        )

    if invalid_groups:
        preview = invalid_groups[:10]

        raise ValueError(
            "Geçersiz ranker grupları bulundu. "
            f"Toplam: {len(invalid_groups)}. "
            f"İlk örnekler: {preview}"
        )

    return {
        "group_count": int(
            dataframe["group_id"].nunique()
        ),
        "company_count": len(
            group_company_ids
        ),
        "groups_by_dataset": dict(
            group_dataset_types
        ),
    }


def build_group_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group_id, group in dataframe.groupby(
        "group_id",
        sort=False,
    ):
        positive_row = group.loc[
            group["label"] == 1
        ].iloc[0]

        expected_rank = int(
            positive_row.get(
                "current_final_rank",
                positive_row.get(
                    "candidate_position",
                    0,
                ),
            )
        )

        if expected_rank <= 0:
            expected_rank = int(
                positive_row.get(
                    "candidate_position",
                    0,
                )
            )

        rows.append(
            {
                "group_id": group_id,
                "dataset_type": str(
                    positive_row["dataset_type"]
                ),
                "expected_company_id": str(
                    positive_row["expected_company_id"]
                ),
                "expected_rank": expected_rank,
                "candidate_count": len(
                    group
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def split_company_disjoint(
    group_summary: pd.DataFrame,
    validation_ratio: float,
    seed: int,
) -> tuple[
    set[str],
    set[str],
    dict[str, Any],
]:
    company_summary_rows: list[
        dict[str, Any]
    ] = []

    for (
        expected_company_id,
        company_groups,
    ) in group_summary.groupby(
        "expected_company_id",
        sort=False,
    ):
        company_summary_rows.append(
            {
                "expected_company_id": (
                    expected_company_id
                ),
                "group_count": len(
                    company_groups
                ),
                "has_hard_positive": bool(
                    (
                        company_groups[
                            "dataset_type"
                        ]
                        == "HARD_POSITIVE"
                    ).any()
                ),
            }
        )

    company_summary = pd.DataFrame(
        company_summary_rows
    )

    company_ids = list(
        company_summary[
            "expected_company_id"
        ]
    )

    if len(company_ids) < 2:
        raise ValueError(
            "Company-disjoint validation için en az "
            "iki farklı şirket gereklidir."
        )

    validation_company_count = int(
        round(
            len(company_ids)
            * validation_ratio
        )
    )

    validation_company_count = max(
        1,
        validation_company_count,
    )

    validation_company_count = min(
        len(company_ids) - 1,
        validation_company_count,
    )

    hard_company_ids = set(
        company_summary.loc[
            company_summary[
                "has_hard_positive"
            ],
            "expected_company_id",
        ]
    )

    best_split: tuple[
        set[str],
        set[str],
    ] | None = None

    random_generator = random.Random(
        seed
    )

    for attempt in range(500):
        shuffled_company_ids = list(
            company_ids
        )

        random_generator.shuffle(
            shuffled_company_ids
        )

        validation_companies = set(
            shuffled_company_ids[
                :validation_company_count
            ]
        )

        training_companies = set(
            shuffled_company_ids[
                validation_company_count:
            ]
        )

        if (
            not validation_companies
            or not training_companies
        ):
            continue

        validation_hard_count = len(
            validation_companies
            & hard_company_ids
        )

        training_hard_count = len(
            training_companies
            & hard_company_ids
        )

        hard_split_required = (
            len(hard_company_ids) >= 2
        )

        if (
            hard_split_required
            and (
                validation_hard_count == 0
                or training_hard_count == 0
            )
        ):
            continue

        best_split = (
            training_companies,
            validation_companies,
        )

        break

    if best_split is None:
        shuffled_company_ids = list(
            company_ids
        )

        random.Random(
            seed
        ).shuffle(
            shuffled_company_ids
        )

        validation_companies = set(
            shuffled_company_ids[
                :validation_company_count
            ]
        )

        training_companies = set(
            shuffled_company_ids[
                validation_company_count:
            ]
        )

    else:
        (
            training_companies,
            validation_companies,
        ) = best_split

    training_group_ids = set(
        group_summary.loc[
            group_summary[
                "expected_company_id"
            ].isin(
                training_companies
            ),
            "group_id",
        ]
    )

    validation_group_ids = set(
        group_summary.loc[
            group_summary[
                "expected_company_id"
            ].isin(
                validation_companies
            ),
            "group_id",
        ]
    )

    company_overlap = (
        training_companies
        & validation_companies
    )

    if company_overlap:
        raise RuntimeError(
            "Eğitim ve doğrulama şirketleri çakışıyor: "
            f"{sorted(company_overlap)}"
        )

    split_summary = {
        "split_strategy": (
            "expected_company_disjoint"
        ),
        "training_company_count": len(
            training_companies
        ),
        "validation_company_count": len(
            validation_companies
        ),
        "training_group_count": len(
            training_group_ids
        ),
        "validation_group_count": len(
            validation_group_ids
        ),
        "company_overlap_count": len(
            company_overlap
        ),
    }

    return (
        training_group_ids,
        validation_group_ids,
        split_summary,
    )


def calculate_group_weights(
    dataframe: pd.DataFrame,
    standard_positive_weight: float,
    hard_positive_weight: float,
    hard_rank_weight_step: float,
) -> pd.Series:
    group_weight_map: dict[
        str,
        float,
    ] = {}

    for group_id, group in dataframe.groupby(
        "group_id",
        sort=False,
    ):
        positive_row = group.loc[
            group["label"] == 1
        ].iloc[0]

        dataset_type = str(
            positive_row["dataset_type"]
        )

        expected_rank = int(
            positive_row.get(
                "current_final_rank",
                positive_row.get(
                    "candidate_position",
                    1,
                ),
            )
        )

        expected_rank = max(
            1,
            expected_rank,
        )

        if dataset_type == "HARD_POSITIVE":
            rank_bonus = (
                max(
                    0,
                    expected_rank - 2,
                )
                * hard_rank_weight_step
            )

            group_weight = (
                hard_positive_weight
                + rank_bonus
            )

        else:
            group_weight = (
                standard_positive_weight
            )

        group_weight_map[
            group_id
        ] = float(
            group_weight
        )

    return dataframe["group_id"].map(
        group_weight_map
    ).astype(np.float64)


def prepare_pool(
    dataframe: pd.DataFrame,
    standard_positive_weight: float,
    hard_positive_weight: float,
    hard_rank_weight_step: float,
) -> tuple[
    pd.DataFrame,
    Pool,
]:
    ordered = dataframe.sort_values(
        by=[
            "group_id",
            "candidate_position",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    feature_frame = ordered.loc[
        :,
        list(
            FEATURE_COLUMNS
        ),
    ].astype(np.float64)

    labels = ordered[
        "label"
    ].astype(np.float64)

    group_ids = ordered[
        "group_id"
    ].astype(str)

    group_weights = calculate_group_weights(
        ordered,
        standard_positive_weight,
        hard_positive_weight,
        hard_rank_weight_step,
    )

    pool = Pool(
        data=feature_frame,
        label=labels,
        group_id=group_ids,
        group_weight=group_weights,
        feature_names=list(
            FEATURE_COLUMNS
        ),
    )

    return (
        ordered,
        pool,
    )


def resolve_task_type(
    requested_task_type: str,
) -> str:
    if requested_task_type in {
        "CPU",
        "GPU",
    }:
        return requested_task_type

    try:
        from catboost.utils import (
            get_gpu_device_count,
        )

        gpu_count = int(
            get_gpu_device_count()
        )

    except Exception:
        gpu_count = 0

    return (
        "GPU"
        if gpu_count > 0
        else "CPU"
    )


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

    return 1.0 / np.log2(
        rank + 1
    )


def evaluate_rankings(
    dataframe: pd.DataFrame,
    prediction_scores: np.ndarray,
) -> dict[str, Any]:
    if len(dataframe) != len(
        prediction_scores
    ):
        raise ValueError(
            "Tahmin sayısı ile doğrulama satırı sayısı eşleşmiyor."
        )

    evaluated = dataframe.copy()

    evaluated[
        "_catboost_prediction"
    ] = prediction_scores

    detail_rows: list[
        dict[str, Any]
    ] = []

    for group_id, group in evaluated.groupby(
        "group_id",
        sort=False,
    ):
        positive_rows = group.loc[
            group["label"] == 1
        ]

        if len(positive_rows) != 1:
            continue

        expected_row = (
            positive_rows.iloc[0]
        )

        expected_company_id = str(
            expected_row[
                "company_id"
            ]
        )

        baseline_order = group.sort_values(
            by=[
                "current_final_rank",
                "candidate_position",
            ],
            ascending=[
                True,
                True,
            ],
            kind="stable",
        )

        model_order = group.sort_values(
            by=[
                "_catboost_prediction",
                "current_final_rank",
            ],
            ascending=[
                False,
                True,
            ],
            kind="stable",
        )

        baseline_company_ids = list(
            baseline_order[
                "company_id"
            ].astype(str)
        )

        model_company_ids = list(
            model_order[
                "company_id"
            ].astype(str)
        )

        baseline_rank = (
            baseline_company_ids.index(
                expected_company_id
            )
            + 1
        )

        model_rank = (
            model_company_ids.index(
                expected_company_id
            )
            + 1
        )

        detail_rows.append(
            {
                "group_id": group_id,
                "dataset_type": str(
                    expected_row[
                        "dataset_type"
                    ]
                ),
                "expected_company_id": (
                    expected_company_id
                ),
                "baseline_rank": (
                    baseline_rank
                ),
                "model_rank": (
                    model_rank
                ),
                "baseline_top1_correct": (
                    baseline_rank == 1
                ),
                "model_top1_correct": (
                    model_rank == 1
                ),
                "improved": (
                    model_rank
                    < baseline_rank
                ),
                "hurt": (
                    model_rank
                    > baseline_rank
                ),
                "unchanged": (
                    model_rank
                    == baseline_rank
                ),
                "winner_changed": (
                    baseline_company_ids[0]
                    != model_company_ids[0]
                ),
            }
        )

    detail_frame = pd.DataFrame(
        detail_rows
    )

    if detail_frame.empty:
        raise ValueError(
            "Doğrulama metriği hesaplanacak grup bulunamadı."
        )

    def summarize(
        subset: pd.DataFrame,
    ) -> dict[str, Any]:
        if subset.empty:
            return {
                "group_count": 0,
                "baseline_top1_accuracy": None,
                "model_top1_accuracy": None,
                "baseline_mrr": None,
                "model_mrr": None,
                "baseline_ndcg": None,
                "model_ndcg": None,
                "improved_count": 0,
                "hurt_count": 0,
                "unchanged_count": 0,
                "winner_changed_count": 0,
            }

        baseline_ranks = list(
            subset[
                "baseline_rank"
            ].astype(int)
        )

        model_ranks = list(
            subset[
                "model_rank"
            ].astype(int)
        )

        return {
            "group_count": len(
                subset
            ),
            "baseline_top1_accuracy": float(
                subset[
                    "baseline_top1_correct"
                ].mean()
            ),
            "model_top1_accuracy": float(
                subset[
                    "model_top1_correct"
                ].mean()
            ),
            "baseline_mrr": float(
                np.mean(
                    [
                        reciprocal_rank(
                            rank
                        )
                        for rank
                        in baseline_ranks
                    ]
                )
            ),
            "model_mrr": float(
                np.mean(
                    [
                        reciprocal_rank(
                            rank
                        )
                        for rank
                        in model_ranks
                    ]
                )
            ),
            "baseline_ndcg": float(
                np.mean(
                    [
                        ndcg_single_relevant(
                            rank
                        )
                        for rank
                        in baseline_ranks
                    ]
                )
            ),
            "model_ndcg": float(
                np.mean(
                    [
                        ndcg_single_relevant(
                            rank
                        )
                        for rank
                        in model_ranks
                    ]
                )
            ),
            "improved_count": int(
                subset[
                    "improved"
                ].sum()
            ),
            "hurt_count": int(
                subset[
                    "hurt"
                ].sum()
            ),
            "unchanged_count": int(
                subset[
                    "unchanged"
                ].sum()
            ),
            "winner_changed_count": int(
                subset[
                    "winner_changed"
                ].sum()
            ),
        }

    return {
        "overall": summarize(
            detail_frame
        ),
        "standard_positive": summarize(
            detail_frame.loc[
                detail_frame[
                    "dataset_type"
                ]
                == "POSITIVE"
            ]
        ),
        "hard_positive": summarize(
            detail_frame.loc[
                detail_frame[
                    "dataset_type"
                ]
                == "HARD_POSITIVE"
            ]
        ),
        "details": detail_rows,
    }


def create_model(
    arguments: argparse.Namespace,
    task_type: str,
) -> CatBoostRanker:
    parameters: dict[str, Any] = {
        "loss_function": "YetiRank",
        "eval_metric": "NDCG:top=10",
        "iterations": (
            arguments.iterations
        ),
        "learning_rate": (
            arguments.learning_rate
        ),
        "depth": arguments.depth,
        "l2_leaf_reg": (
            arguments.l2_leaf_reg
        ),
        "random_seed": (
            arguments.seed
        ),
        "task_type": task_type,
        "allow_writing_files": False,
        "verbose": (
            arguments.metric_period
        ),
    }

    if task_type == "GPU":
        parameters[
            "devices"
        ] = arguments.devices

    else:
        parameters[
            "thread_count"
        ] = -1

    return CatBoostRanker(
        **parameters
    )


def train_model(
    arguments: argparse.Namespace,
    training_pool: Pool,
    validation_pool: Pool,
) -> tuple[
    CatBoostRanker,
    str,
    str | None,
]:
    resolved_task_type = resolve_task_type(
        arguments.task_type
    )

    fallback_reason: str | None = None

    print(
        "CatBoost task type: "
        f"{resolved_task_type}"
    )

    model = create_model(
        arguments,
        resolved_task_type,
    )

    try:
        model.fit(
            training_pool,
            eval_set=validation_pool,
            use_best_model=True,
            early_stopping_rounds=(
                arguments.early_stopping_rounds
            ),
            verbose=(
                arguments.metric_period
            ),
        )

        return (
            model,
            resolved_task_type,
            fallback_reason,
        )

    except CatBoostError as exc:
        if (
            arguments.task_type != "AUTO"
            or resolved_task_type != "GPU"
        ):
            raise

        fallback_reason = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "GPU eğitimi başarısız oldu. "
            "CPU ile tekrar deneniyor."
        )

        model = create_model(
            arguments,
            "CPU",
        )

        model.fit(
            training_pool,
            eval_set=validation_pool,
            use_best_model=True,
            early_stopping_rounds=(
                arguments.early_stopping_rounds
            ),
            verbose=(
                arguments.metric_period
            ),
        )

        return (
            model,
            "CPU",
            fallback_reason,
        )


def backup_existing_file(
    path: Path,
) -> Path | None:
    if not path.exists():
        return None

    timestamp = datetime.now(
        UTC
    ).strftime(
        "%Y%m%dT%H%M%SZ"
    )

    backup_path = path.with_name(
        f"{path.stem}.backup_{timestamp}"
        f"{path.suffix}"
    )

    shutil.copy2(
        path,
        backup_path,
    )

    return backup_path


def sha256_file(
    path: Path,
) -> str:
    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as file_handle:
        for chunk in iter(
            lambda: file_handle.read(
                1024 * 1024
            ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def calculate_feature_importance(
    model: CatBoostRanker,
    importance_pool: Pool,
) -> list[dict[str, Any]]:
    """
    Ranking modeli için LossFunctionChange tabanlı
    feature importance değerlerini hesaplar.

    CatBoostRanker'da LossFunctionChange hesaplaması
    bir veri havuzu gerektirir.
    """

    values = model.get_feature_importance(
        data=importance_pool,
        type="LossFunctionChange",
        prettified=False,
    )

    if len(values) != len(
        FEATURE_COLUMNS
    ):
        raise ValueError(
            "Feature importance sayısı ile feature "
            "kolonu sayısı eşleşmiyor. "
            f"Importance: {len(values)}, "
            f"Feature: {len(FEATURE_COLUMNS)}"
        )

    importance_rows = [
        {
            "feature": feature_name,
            "importance": float(
                importance
            ),
        }
        for feature_name, importance
        in zip(
            FEATURE_COLUMNS,
            values,
            strict=True,
        )
    ]

    return sorted(
        importance_rows,
        key=lambda item: item[
            "importance"
        ],
        reverse=True,
    )

def count_groups_by_dataset(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    group_frame = (
        dataframe[
            [
                "group_id",
                "dataset_type",
            ]
        ]
        .drop_duplicates(
            subset=[
                "group_id",
            ]
        )
    )

    return {
        str(dataset_type): int(
            count
        )
        for dataset_type, count
        in group_frame[
            "dataset_type"
        ].value_counts().items()
    }


def main() -> None:
    arguments = parse_arguments()

    validate_arguments(
        arguments
    )

    dataframe = read_dataset(
        arguments.input
    )

    eligible_dataframe = filter_eligible_data(
        dataframe
    )

    dataset_validation = validate_groups(
        eligible_dataframe
    )

    group_summary = build_group_summary(
        eligible_dataframe
    )

    (
        training_group_ids,
        validation_group_ids,
        split_summary,
    ) = split_company_disjoint(
        group_summary,
        arguments.validation_ratio,
        arguments.seed,
    )

    training_dataframe = (
        eligible_dataframe.loc[
            eligible_dataframe[
                "group_id"
            ].isin(
                training_group_ids
            )
        ]
        .copy()
    )

    validation_dataframe = (
        eligible_dataframe.loc[
            eligible_dataframe[
                "group_id"
            ].isin(
                validation_group_ids
            )
        ]
        .copy()
    )

    if training_dataframe.empty:
        raise ValueError(
            "Eğitim veri seti boş kaldı."
        )

    if validation_dataframe.empty:
        raise ValueError(
            "Doğrulama veri seti boş kaldı."
        )

    (
        ordered_training_dataframe,
        training_pool,
    ) = prepare_pool(
        training_dataframe,
        arguments.standard_positive_weight,
        arguments.hard_positive_weight,
        arguments.hard_rank_weight_step,
    )

    (
        ordered_validation_dataframe,
        validation_pool,
    ) = prepare_pool(
        validation_dataframe,
        arguments.standard_positive_weight,
        arguments.hard_positive_weight,
        arguments.hard_rank_weight_step,
    )

    print(
        "Candidate feature veri seti hazır."
    )

    print(
        "Toplam ranker-uygun grup: "
        f"{dataset_validation['group_count']}"
    )

    print(
        "Toplam farklı şirket: "
        f"{dataset_validation['company_count']}"
    )

    print(
        "Grup dağılımı: "
        f"{dataset_validation['groups_by_dataset']}"
    )

    print(
        "Eğitim grup sayısı: "
        f"{len(training_group_ids)}"
    )

    print(
        "Doğrulama grup sayısı: "
        f"{len(validation_group_ids)}"
    )

    print(
        "Eğitim grup dağılımı: "
        f"{count_groups_by_dataset(training_dataframe)}"
    )

    print(
        "Doğrulama grup dağılımı: "
        f"{count_groups_by_dataset(validation_dataframe)}"
    )

    print(
        "Hard-positive grup ağırlığı: "
        f"{arguments.hard_positive_weight}"
    )

    model, actual_task_type, fallback_reason = (
        train_model(
            arguments,
            training_pool,
            validation_pool,
        )
    )

    validation_predictions = np.asarray(
        model.predict(
            validation_pool
        ),
        dtype=np.float64,
    ).reshape(-1)

    evaluation = evaluate_rankings(
        ordered_validation_dataframe,
        validation_predictions,
    )

    arguments.model_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments.metadata_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_backup_path = backup_existing_file(
        arguments.model_output
    )

    metadata_backup_path = backup_existing_file(
        arguments.metadata_output
    )

    model.save_model(
        str(
            arguments.model_output
        )
    )

    feature_importance = (
        calculate_feature_importance(
            model=model,
            importance_pool=training_pool,
        )
    )

    best_iteration = int(
        model.get_best_iteration()
    )

    best_score = model.get_best_score()

    metadata = {
        "model_type": "CatBoostRanker",
        "model_version": "2.0.0-hard-positive",
        "created_at": datetime.now(
            UTC
        ).isoformat(),
        "input_dataset": str(
            arguments.input.resolve()
        ),
        "model_path": str(
            arguments.model_output.resolve()
        ),
        "model_sha256": sha256_file(
            arguments.model_output
        ),
        "feature_columns": list(
            FEATURE_COLUMNS
        ),
        "feature_count": len(
            FEATURE_COLUMNS
        ),
        "excluded_feature_families": [
            "label",
            "company identifiers",
            "query and company text",
            "current_final_rank",
            "catboost shadow scores",
            "catboost shadow ranks",
            "catboost shadow margins",
        ],
        "dataset": {
            "eligible_candidate_rows": len(
                eligible_dataframe
            ),
            "eligible_group_count": int(
                eligible_dataframe[
                    "group_id"
                ].nunique()
            ),
            "eligible_company_count": int(
                eligible_dataframe[
                    "expected_company_id"
                ].nunique()
            ),
            "groups_by_dataset": (
                dataset_validation[
                    "groups_by_dataset"
                ]
            ),
            "training_candidate_rows": len(
                ordered_training_dataframe
            ),
            "validation_candidate_rows": len(
                ordered_validation_dataframe
            ),
            "training_groups_by_dataset": (
                count_groups_by_dataset(
                    training_dataframe
                )
            ),
            "validation_groups_by_dataset": (
                count_groups_by_dataset(
                    validation_dataframe
                )
            ),
        },
        "split": split_summary,
        "group_weights": {
            "standard_positive_weight": (
                arguments.standard_positive_weight
            ),
            "hard_positive_weight": (
                arguments.hard_positive_weight
            ),
            "hard_rank_weight_step": (
                arguments.hard_rank_weight_step
            ),
        },
        "training": {
            "requested_task_type": (
                arguments.task_type
            ),
            "actual_task_type": (
                actual_task_type
            ),
            "gpu_fallback_reason": (
                fallback_reason
            ),
            "devices": arguments.devices,
            "iterations": (
                arguments.iterations
            ),
            "learning_rate": (
                arguments.learning_rate
            ),
            "depth": arguments.depth,
            "l2_leaf_reg": (
                arguments.l2_leaf_reg
            ),
            "early_stopping_rounds": (
                arguments.early_stopping_rounds
            ),
            "loss_function": "YetiRank",
            "eval_metric": "NDCG:top=10",
            "seed": arguments.seed,
            "best_iteration": (
                best_iteration
            ),
            "best_score": best_score,
        },
        "validation_metrics": {
            "overall": evaluation[
                "overall"
            ],
            "standard_positive": evaluation[
                "standard_positive"
            ],
            "hard_positive": evaluation[
                "hard_positive"
            ],
        },
        "validation_details": (
            evaluation[
                "details"
            ]
        ),
        "feature_importance": (
            feature_importance
        ),
        "backup_paths": {
            "previous_model": (
                str(
                    model_backup_path
                )
                if model_backup_path
                else None
            ),
            "previous_metadata": (
                str(
                    metadata_backup_path
                )
                if metadata_backup_path
                else None
            ),
        },
    }

    arguments.metadata_output.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    overall_metrics = evaluation[
        "overall"
    ]

    standard_metrics = evaluation[
        "standard_positive"
    ]

    hard_metrics = evaluation[
        "hard_positive"
    ]

    print()
    print(
        "CatBoostRanker eğitimi tamamlandı."
    )

    print(
        "Best iteration: "
        f"{best_iteration}"
    )

    print(
        "Doğrulama genel mevcut Top-1: "
        f"{overall_metrics['baseline_top1_accuracy']:.2%}"
    )

    print(
        "Doğrulama genel CatBoost Top-1: "
        f"{overall_metrics['model_top1_accuracy']:.2%}"
    )

    print(
        "Doğrulama standart pozitif mevcut Top-1: "
        + (
            f"{standard_metrics['baseline_top1_accuracy']:.2%}"
            if standard_metrics[
                "baseline_top1_accuracy"
            ] is not None
            else "N/A"
        )
    )

    print(
        "Doğrulama standart pozitif CatBoost Top-1: "
        + (
            f"{standard_metrics['model_top1_accuracy']:.2%}"
            if standard_metrics[
                "model_top1_accuracy"
            ] is not None
            else "N/A"
        )
    )

    print(
        "Doğrulama hard-positive mevcut Top-1: "
        + (
            f"{hard_metrics['baseline_top1_accuracy']:.2%}"
            if hard_metrics[
                "baseline_top1_accuracy"
            ] is not None
            else "N/A"
        )
    )

    print(
        "Doğrulama hard-positive CatBoost Top-1: "
        + (
            f"{hard_metrics['model_top1_accuracy']:.2%}"
            if hard_metrics[
                "model_top1_accuracy"
            ] is not None
            else "N/A"
        )
    )

    print(
        "İyileşen doğrulama grubu: "
        f"{overall_metrics['improved_count']}"
    )

    print(
        "Kötüleşen doğrulama grubu: "
        f"{overall_metrics['hurt_count']}"
    )

    print(
        "Değişmeyen doğrulama grubu: "
        f"{overall_metrics['unchanged_count']}"
    )

    print(
        "En önemli 10 feature:"
    )

    for item in feature_importance[:10]:
        print(
            f"  {item['feature']}: "
            f"{item['importance']:.6f}"
        )

    print(
        f"Model: {arguments.model_output}"
    )

    print(
        f"Metadata: {arguments.metadata_output}"
    )


if __name__ == "__main__":
    main()