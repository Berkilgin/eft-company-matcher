from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]
)

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "benchmark_summary.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "benchmark"
    / "results"
    / "benchmark_results.csv"
)


st.set_page_config(
    page_title="Model Metrikleri",
    layout="wide",
)


st.title(
    "Model Benchmark Sonuçları"
)

st.caption(
    "Embedding, reranker ve karar "
    "mekanizmasının uçtan uca "
    "performansını gösterir."
)


def as_percent(
    value: float | None,
) -> str:
    if value is None:
        return "N/A"

    return f"%{value * 100:.2f}"


def boolean_series(
    series: pd.Series,
) -> pd.Series:
    return (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(
            {
                "true",
                "1",
                "yes",
            }
        )
    )


if not SUMMARY_PATH.exists():
    st.warning(
        "Benchmark özeti bulunamadı. "
        "Önce run_benchmark.py "
        "scriptini çalıştırın."
    )
    st.stop()


summary = json.loads(
    SUMMARY_PATH.read_text(
        encoding="utf-8"
    )
)

overall = summary[
    "overall"
]


metric_column_1, metric_column_2, (
    metric_column_3
), metric_column_4 = st.columns(4)


with metric_column_1:
    st.metric(
        "Top-1 Accuracy",
        as_percent(
            overall[
                "top1_accuracy"
            ]
        ),
    )


with metric_column_2:
    st.metric(
        "Candidate Recall@5",
        as_percent(
            overall[
                "candidate_recall_at_5"
            ]
        ),
    )


with metric_column_3:
    st.metric(
        "Candidate Recall@10",
        as_percent(
            overall[
                "candidate_recall_at_10"
            ]
        ),
    )


with metric_column_4:
    st.metric(
        "MRR",
        (
            f'{overall["mean_reciprocal_rank"]:.4f}'
            if overall[
                "mean_reciprocal_rank"
            ] is not None
            else "N/A"
        ),
    )


metric_column_5, metric_column_6, (
    metric_column_7
), metric_column_8 = st.columns(4)


with metric_column_5:
    st.metric(
        "AUTO_MATCH Precision",
        as_percent(
            overall[
                "auto_match_precision"
            ]
        ),
    )


with metric_column_6:
    st.metric(
        "AUTO_MATCH Coverage",
        as_percent(
            overall[
                "auto_match_coverage"
            ]
        ),
    )


with metric_column_7:
    st.metric(
        "False Match Rate",
        as_percent(
            overall[
                "false_match_rate"
            ]
        ),
    )


with metric_column_8:
    p95_latency = overall[
        "p95_latency_ms"
    ]

    st.metric(
        "P95 Gecikme",
        (
            f"{p95_latency:.1f} ms"
            if p95_latency is not None
            else "N/A"
        ),
    )


st.divider()
st.subheader(
    "Zorluk seviyesine göre sonuçlar"
)


difficulty_rows = []

for difficulty, metrics in (
    summary[
        "by_difficulty"
    ].items()
):
    difficulty_rows.append(
        {
            "Zorluk": difficulty,
            "Sorgu": metrics[
                "total_queries"
            ],
            "Top-1": as_percent(
                metrics[
                    "top1_accuracy"
                ]
            ),
            "Recall@5": as_percent(
                metrics[
                    "candidate_recall_at_5"
                ]
            ),
            "Recall@10": as_percent(
                metrics[
                    "candidate_recall_at_10"
                ]
            ),
            "MRR": metrics[
                "mean_reciprocal_rank"
            ],
            "AUTO Precision": (
                as_percent(
                    metrics[
                        "auto_match_precision"
                    ]
                )
            ),
            "UNKNOWN": as_percent(
                metrics[
                    "unknown_rate"
                ]
            ),
        }
    )


st.dataframe(
    pd.DataFrame(
        difficulty_rows
    ),
    hide_index=True,
    use_container_width=True,
)


st.divider()
st.subheader(
    "Hata analizi"
)


if not RESULTS_PATH.exists():
    st.info(
        "Detay sonuç dosyası "
        "bulunamadı."
    )
    st.stop()


results = pd.read_csv(
    RESULTS_PATH
)


results[
    "top1_correct_bool"
] = boolean_series(
    results[
        "top1_correct"
    ]
)


if (
    "ambiguous_query"
    in results.columns
):
    results[
        "ambiguous_query_bool"
    ] = boolean_series(
        results[
            "ambiguous_query"
        ]
    )
else:
    results[
        "ambiguous_query_bool"
    ] = False


results[
    "true_rank_numeric"
] = pd.to_numeric(
    results[
        "true_rank"
    ],
    errors="coerce",
)


results[
    "final_score"
] = pd.to_numeric(
    results[
        "final_score"
    ],
    errors="coerce",
)


results[
    "score_margin"
] = pd.to_numeric(
    results[
        "score_margin"
    ],
    errors="coerce",
)


def determine_error_type(
    row: pd.Series,
) -> str:
    if (
        row.get(
            "decision"
        )
        == "AUTO_MATCH"
        and not row[
            "top1_correct_bool"
        ]
    ):
        return (
            "KRİTİK: YANLIŞ AUTO"
        )

    if row[
        "ambiguous_query_bool"
    ]:
        return "BELİRSİZ SORGU"

    if pd.isna(
        row[
            "true_rank_numeric"
        ]
    ):
        return (
            "ADAY İLK 10'DA YOK"
        )

    if (
        row[
            "true_rank_numeric"
        ]
        == 2
    ):
        return (
            "YAKIN HATA: 2. SIRA"
        )

    return "SIRALAMA HATASI"


incorrect_results = results[
    ~results[
        "top1_correct_bool"
    ]
].copy()


incorrect_results[
    "error_type"
] = incorrect_results.apply(
    determine_error_type,
    axis=1,
)


critical_count = int(
    (
        incorrect_results[
            "error_type"
        ]
        == "KRİTİK: YANLIŞ AUTO"
    ).sum()
)


candidate_miss_count = int(
    (
        incorrect_results[
            "error_type"
        ]
        == "ADAY İLK 10'DA YOK"
    ).sum()
)


rank_two_count = int(
    (
        incorrect_results[
            "error_type"
        ]
        == "YAKIN HATA: 2. SIRA"
    ).sum()
)


ambiguous_count = int(
    incorrect_results[
        "ambiguous_query_bool"
    ].sum()
)


summary_column_1, summary_column_2, (
    summary_column_3
), summary_column_4 = st.columns(4)


with summary_column_1:
    st.metric(
        "Yanlış AUTO_MATCH",
        critical_count,
    )


with summary_column_2:
    st.metric(
        "İlk 10 adayda yok",
        candidate_miss_count,
    )


with summary_column_3:
    st.metric(
        "Doğru şirket 2. sırada",
        rank_two_count,
    )


with summary_column_4:
    st.metric(
        "Belirsiz hata sorgusu",
        ambiguous_count,
    )


filter_column_1, filter_column_2, (
    filter_column_3
), filter_column_4 = st.columns(4)


with filter_column_1:
    selected_error_types = (
        st.multiselect(
            "Hata türü",
            options=sorted(
                incorrect_results[
                    "error_type"
                ]
                .dropna()
                .unique()
            ),
            default=[],
        )
    )


with filter_column_2:
    selected_decisions = (
        st.multiselect(
            "Karar",
            options=sorted(
                incorrect_results[
                    "decision"
                ]
                .dropna()
                .unique()
            ),
            default=[],
        )
    )


with filter_column_3:
    selected_difficulties = (
        st.multiselect(
            "Zorluk",
            options=sorted(
                incorrect_results[
                    "difficulty"
                ]
                .dropna()
                .unique()
            ),
            default=[],
        )
    )


with filter_column_4:
    selected_corruptions = (
        st.multiselect(
            "Bozma yöntemi",
            options=sorted(
                incorrect_results[
                    "corruption_type"
                ]
                .dropna()
                .unique()
            ),
            default=[],
        )
    )


include_ambiguous = st.checkbox(
    "Belirsiz sorguları hata "
    "tablosunda göster",
    value=True,
)


filtered_results = (
    incorrect_results.copy()
)


if not include_ambiguous:
    filtered_results = (
        filtered_results[
            ~filtered_results[
                "ambiguous_query_bool"
            ]
        ]
    )


if selected_error_types:
    filtered_results = (
        filtered_results[
            filtered_results[
                "error_type"
            ].isin(
                selected_error_types
            )
        ]
    )


if selected_decisions:
    filtered_results = (
        filtered_results[
            filtered_results[
                "decision"
            ].isin(
                selected_decisions
            )
        ]
    )


if selected_difficulties:
    filtered_results = (
        filtered_results[
            filtered_results[
                "difficulty"
            ].isin(
                selected_difficulties
            )
        ]
    )


if selected_corruptions:
    filtered_results = (
        filtered_results[
            filtered_results[
                "corruption_type"
            ].isin(
                selected_corruptions
            )
        ]
    )


error_priority = {
    "KRİTİK: YANLIŞ AUTO": 0,
    "ADAY İLK 10'DA YOK": 1,
    "YAKIN HATA: 2. SIRA": 2,
    "SIRALAMA HATASI": 3,
    "BELİRSİZ SORGU": 4,
}


filtered_results[
    "error_priority"
] = filtered_results[
    "error_type"
].map(
    error_priority
).fillna(99)


filtered_results = (
    filtered_results.sort_values(
        by=[
            "error_priority",
            "final_score",
        ],
        ascending=[
            True,
            False,
        ],
    )
)


st.caption(
    f"{len(filtered_results)} hata "
    "kaydı gösteriliyor."
)


display_column_mapping = {
    "error_type": "Hata türü",
    "query_text": "Sorgu",
    "normalized_query": (
        "Normalize sorgu"
    ),
    "legal_name": (
        "Beklenen şirket"
    ),
    "predicted_legal_name": (
        "Tahmin edilen şirket"
    ),
    "true_rank_numeric": (
        "Gerçek sıra"
    ),
    "decision": "Karar",
    "final_score": (
        "Nihai skor"
    ),
    "retrieval_score": (
        "Retrieval"
    ),
    "reranker_score": (
        "Reranker"
    ),
    "score_margin": (
        "Skor farkı"
    ),
    "difficulty": "Zorluk",
    "corruption_type": (
        "Bozma yöntemi"
    ),
}


available_columns = [
    column
    for column
    in display_column_mapping
    if column
    in filtered_results.columns
]


display_results = filtered_results[
    available_columns
].rename(
    columns={
        column: (
            display_column_mapping[
                column
            ]
        )
        for column
        in available_columns
    }
)


st.dataframe(
    display_results,
    hide_index=True,
    use_container_width=True,
    height=650,
    column_config={
        "Hata türü": (
            st.column_config.TextColumn(
                width="medium",
            )
        ),
        "Sorgu": (
            st.column_config.TextColumn(
                width="large",
            )
        ),
        "Normalize sorgu": (
            st.column_config.TextColumn(
                width="medium",
            )
        ),
        "Beklenen şirket": (
            st.column_config.TextColumn(
                width="large",
            )
        ),
        "Tahmin edilen şirket": (
            st.column_config.TextColumn(
                width="large",
            )
        ),
        "Nihai skor": (
            st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        ),
        "Retrieval": (
            st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        ),
        "Reranker": (
            st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        ),
        "Skor farkı": (
            st.column_config.ProgressColumn(
                min_value=0.0,
                max_value=1.0,
                format="%.3f",
            )
        ),
    },
)


st.download_button(
    label=(
        "Filtrelenmiş hataları CSV indir"
    ),
    data=display_results.to_csv(
        index=False,
    ).encode(
        "utf-8-sig"
    ),
    file_name=(
        "filtered_benchmark_errors.csv"
    ),
    mime="text/csv",
)


st.divider()


with st.expander(
    "Ham benchmark özeti"
):
    st.json(summary)