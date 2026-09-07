from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st


FRONTEND_DIRECTORY = Path(__file__).resolve().parents[1]

if str(FRONTEND_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIRECTORY))


from services.api_client import (  # noqa: E402
    BackendConnectionError,
    match_eft_description,
)


st.set_page_config(
    page_title="EFT Eşleştirme",
    page_icon="🏦",
    layout="wide",
)


st.title("EFT Şirket Eşleştirme")

st.caption(
    "EFT açıklamasını normalize eder, şirket adaylarını bulur "
    "ve Qwen3 Reranker ile yeniden sıralar."
)


with st.sidebar:
    st.header("Ayarlar")

    backend_url = st.text_input(
        "Backend adresi",
        value="http://127.0.0.1:8000",
        key="matching_backend_url",
    )

    candidate_limit = st.slider(
        "Aday sayısı",
        min_value=1,
        max_value=20,
        value=10,
    )


eft_description = st.text_area(
    "EFT açıklaması",
    value="TUPRASS IZM FATURA ODEMESI",
    height=130,
    placeholder="EFT açıklamasını buraya yazın...",
)


match_button = st.button(
    "Şirketi bul",
    type="primary",
    use_container_width=True,
)


if match_button:
    if not eft_description.strip():
        st.warning("EFT açıklaması boş bırakılamaz.")
        st.stop()

    try:
        with st.spinner("Şirket adayları hesaplanıyor..."):
            result = match_eft_description(
                text=eft_description,
                limit=candidate_limit,
                backend_url=backend_url,
            )

    except BackendConnectionError as exc:
        st.error(str(exc))
        st.stop()

    decision = result["decision"]
    best_candidate = result["best_candidate"]

    decision_labels = {
        "AUTO_MATCH": "Otomatik eşleşme",
        "MANUAL_REVIEW": "Manuel inceleme gerekli",
        "UNKNOWN": "Şirket bulunamadı",
    }

    st.subheader("Eşleştirme sonucu")

    ranking_mode = result.get(
        "ranking_mode",
        "retrieval_only",
    )

    if result.get("reranker_applied", False):
        st.info(
            "Qwen3 Reranker aktif. "
            f'{result.get("reranked_candidate_count", 0)} '
            "aday yeniden sıralandı."
        )
    else:
        st.info(
            f"Reranker uygulanmadı. Mod: {ranking_mode}"
        )

    if result.get("ranking_error"):
        st.warning(
            "Reranker hatası: "
            f'{result["ranking_error"]}'
        )

    if decision == "AUTO_MATCH":
        st.success(
            decision_labels.get(
                decision,
                decision,
            )
        )

    elif decision == "MANUAL_REVIEW":
        st.warning(
            decision_labels.get(
                decision,
                decision,
            )
        )

    else:
        st.error(
            decision_labels.get(
                decision,
                decision,
            )
        )

    if best_candidate:
        company_column, score_column, margin_column = (
            st.columns(3)
        )

        with company_column:
            st.metric(
                label="En iyi aday",
                value=(
                    best_candidate["brand_name"]
                    or best_candidate["legal_name"]
                ),
            )

        with score_column:
            final_score = best_candidate.get(
                "final_score",
                best_candidate.get(
                    "hybrid_score",
                    0.0,
                ),
            )

            st.metric(
                label="Nihai skor",
                value=f"%{final_score * 100:.2f}",
            )

        with margin_column:
            st.metric(
                label="İlk iki aday farkı",
                value=(
                    f'%{result["score_margin"] * 100:.2f}'
                ),
            )

        st.markdown(
            f'**Resmî unvan:** '
            f'{best_candidate["legal_name"]}'
        )

        st.markdown(
            f'**Eşleşen alias:** '
            f'`{best_candidate["matched_alias"]}`'
        )

        st.markdown(
            f'**Şehir / sektör:** '
            f'{best_candidate["city"]} / '
            f'{best_candidate["sector"]}'
        )

        score_detail_1, score_detail_2, score_detail_3 = (
            st.columns(3)
        )

        with score_detail_1:
            st.metric(
                "Retrieval skoru",
                f'%{best_candidate.get("retrieval_score", 0) * 100:.2f}',
            )

        with score_detail_2:
            st.metric(
                "Reranker skoru",
                f'%{best_candidate.get("reranker_score", 0) * 100:.2f}',
            )

        with score_detail_3:
            st.metric(
                "Embedding skoru",
                f'%{best_candidate.get("embedding_score", 0) * 100:.2f}',
            )

    st.divider()

    st.subheader("Normalize edilmiş sorgu")

    normalization = result["normalization"]

    normalization_column_1, normalization_column_2 = (
        st.columns(2)
    )

    with normalization_column_1:
        st.markdown("**Ham açıklama**")
        st.write(normalization["raw_text"])

    with normalization_column_2:
        st.markdown("**Çekirdek metin**")
        st.code(
            result["query_text"]
            or "Çekirdek metin üretilemedi."
        )

    st.divider()

    st.subheader("Şirket adayları")

    if result["candidates"]:
        candidate_rows = []

        for rank, candidate in enumerate(
            result["candidates"],
            start=1,
        ):
            candidate_rows.append(
                {
                    "Sıra": rank,
                    "Şirket": candidate["legal_name"],
                    "Marka": candidate["brand_name"],
                    "Eşleşen alias": (
                        candidate["matched_alias"]
                    ),
                    "Exact": (
                        "Evet"
                        if candidate["exact_match"]
                        else "Hayır"
                    ),
                    "TF-IDF": round(
                        candidate["char_tfidf_score"],
                        4,
                    ),
                    "Embedding": round(
                        candidate.get(
                            "embedding_score",
                            0.0,
                        ),
                        4,
                    ),
                    "WRatio": round(
                        candidate["fuzzy_wratio_score"],
                        4,
                    ),
                    "Retrieval": round(
                        candidate.get(
                            "retrieval_score",
                            candidate.get(
                                "hybrid_score",
                                0.0,
                            ),
                        ),
                        4,
                    ),
                    "Reranker": round(
                        candidate.get(
                            "reranker_score",
                            0.0,
                        ),
                        4,
                    ),
                    "Nihai": round(
                        candidate.get(
                            "final_score",
                            candidate.get(
                                "hybrid_score",
                                0.0,
                            ),
                        ),
                        4,
                    ),
                    "Reranked": (
                        "Evet"
                        if candidate.get(
                            "reranked",
                            False,
                        )
                        else "Hayır"
                    ),
                }
            )

        candidate_dataframe = pd.DataFrame(
            candidate_rows
        )

        st.dataframe(
            candidate_dataframe,
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Şirket adayı bulunamadı.")

    st.divider()

    st.subheader("İndeks durumu")

    index_stats = result["index_stats"]

    stats_column_1, stats_column_2, stats_column_3 = (
        st.columns(3)
    )

    with stats_column_1:
        st.metric(
            "Şirket sayısı",
            index_stats["company_count"],
        )

    with stats_column_2:
        st.metric(
            "Alias sayısı",
            index_stats["alias_count"],
        )

    with stats_column_3:
        st.metric(
            "Embedding",
            (
                "Aktif"
                if index_stats.get(
                    "embedding_enabled",
                    False,
                )
                else "Pasif"
            ),
        )

    with st.expander("Ham API sonucu"):
        st.json(result)