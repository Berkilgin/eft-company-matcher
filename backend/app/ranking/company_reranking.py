from __future__ import annotations

import math
from typing import Any, Protocol, Sequence


class RerankerProvider(Protocol):
    is_loaded: bool

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        ...


class CatBoostRankerProvider(Protocol):
    is_loaded: bool

    def predict_raw_scores(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[float]:
        ...


def apply_company_reranking(
    result: dict[str, Any],
    reranker_model: RerankerProvider | None,
    rerank_limit: int = 10,
    catboost_ranker_model: CatBoostRankerProvider | None = None,
) -> dict[str, Any]:
    """
    Retrieval adaylarını Qwen3 Reranker ile yeniden sıralar.

    CatBoost modeli verilmişse shadow mode çalıştırılır.
    Shadow mode yalnızca alternatif CatBoost sırasını
    hesaplar; gerçek aday sırasını ve eşleştirme kararını
    değiştirmez.
    """

    candidates = result.get(
        "candidates",
        [],
    )

    query_text = str(
        result.get(
            "query_text",
            "",
        )
    ).strip()

    query_length = _calculate_query_length(
        query_text
    )

    query_token_count = len(
        query_text.split()
    )

    for candidate in candidates:
        retrieval_score = _clamp_score(
            candidate.get(
                "hybrid_score",
                0.0,
            )
        )

        lexical_evidence = (
            _calculate_lexical_evidence(
                candidate
            )
        )

        candidate["retrieval_score"] = round(
            retrieval_score,
            6,
        )

        candidate["reranker_score"] = 0.0

        candidate[
            "adjusted_reranker_score"
        ] = 0.0

        candidate["lexical_evidence"] = round(
            lexical_evidence,
            6,
        )

        candidate["retrieval_weight"] = 1.0
        candidate["reranker_weight"] = 0.0
        candidate["reranker_gap"] = 0.0

        candidate["query_length"] = (
            query_length
        )

        candidate["query_token_count"] = (
            query_token_count
        )

        candidate["final_score"] = round(
            retrieval_score,
            6,
        )

        candidate["reranked"] = False

        # CatBoost shadow-mode alanları.
        candidate["retrieval_rank"] = 0
        candidate["current_final_rank"] = 0

        candidate[
            "candidate_reranker_gap"
        ] = 0.0

        candidate[
            "base_reranker_weight"
        ] = 0.0

        candidate["catboost_raw_score"] = 0.0

        candidate[
            "catboost_normalized_score"
        ] = 0.0

        candidate["catboost_rank"] = 0

        candidate[
            "catboost_shadow_top1"
        ] = False

        candidate[
            "catboost_shadow_order_delta"
        ] = 0

    _assign_retrieval_ranks(
        candidates
    )

    result["ranking_mode"] = (
        "retrieval_only"
    )

    result["reranker_applied"] = False
    result["reranked_candidate_count"] = 0
    result["ranking_error"] = None

    result["query_length"] = (
        query_length
    )

    result["query_token_count"] = (
        query_token_count
    )

    result["reranker_gap"] = 0.0

    result["base_reranker_weight"] = 0.0

    _initialize_catboost_shadow_result(
        result
    )

    if not candidates:
        result["best_candidate"] = None
        result["decision"] = "UNKNOWN"
        result["score_margin"] = 0.0

        result["catboost_shadow_status"] = (
            "skipped_no_candidates"
        )

        return result

    if (
        reranker_model is None
        or not reranker_model.is_loaded
    ):
        _sort_and_update_decision(
            result
        )

        result["catboost_shadow_status"] = (
            "skipped_reranker_not_applied"
        )

        return result

    actual_limit = min(
        max(
            rerank_limit,
            1,
        ),
        len(candidates),
    )

    candidates_to_rerank = candidates[
        :actual_limit
    ]

    reranker_query = (
        _build_reranker_query(
            result
        )
    )

    reranker_documents = [
        _build_candidate_document(
            candidate
        )
        for candidate in candidates_to_rerank
    ]

    try:
        raw_reranker_scores = (
            reranker_model.score(
                query=reranker_query,
                documents=reranker_documents,
            )
        )

        if (
            len(raw_reranker_scores)
            != actual_limit
        ):
            raise RuntimeError(
                "Reranker skor sayısı aday "
                "sayısıyla eşleşmiyor."
            )

        reranker_scores = [
            _clamp_score(score)
            for score in raw_reranker_scores
        ]

        reranker_gap = (
            _calculate_reranker_gap(
                reranker_scores
            )
        )

        base_reranker_weight = (
            _determine_base_reranker_weight(
                query_length=query_length,
                query_token_count=(
                    query_token_count
                ),
                reranker_gap=reranker_gap,
            )
        )

        result["reranker_gap"] = round(
            reranker_gap,
            6,
        )

        result[
            "base_reranker_weight"
        ] = round(
            base_reranker_weight,
            6,
        )

        for candidate in candidates:
            candidate[
                "candidate_reranker_gap"
            ] = round(
                reranker_gap,
                6,
            )

            candidate[
                "base_reranker_weight"
            ] = round(
                base_reranker_weight,
                6,
            )

        for candidate, reranker_score in zip(
            candidates_to_rerank,
            reranker_scores,
            strict=True,
        ):
            retrieval_score = float(
                candidate[
                    "retrieval_score"
                ]
            )

            lexical_evidence = float(
                candidate[
                    "lexical_evidence"
                ]
            )

            adjusted_reranker_score = (
                _adjust_reranker_score(
                    reranker_score=(
                        reranker_score
                    ),
                    lexical_evidence=(
                        lexical_evidence
                    ),
                )
            )

            reranker_weight = (
                _determine_candidate_reranker_weight(
                    candidate=candidate,
                    base_reranker_weight=(
                        base_reranker_weight
                    ),
                    lexical_evidence=(
                        lexical_evidence
                    ),
                )
            )

            retrieval_weight = (
                1.0
                - reranker_weight
            )

            if candidate.get(
                "unique_exact_match",
                False,
            ):
                retrieval_weight = 1.0
                reranker_weight = 0.0

                adjusted_reranker_score = (
                    reranker_score
                )

                final_score = 1.0

            else:
                final_score = (
                    retrieval_weight
                    * retrieval_score
                    + reranker_weight
                    * adjusted_reranker_score
                )

            candidate["reranker_score"] = round(
                reranker_score,
                6,
            )

            candidate[
                "adjusted_reranker_score"
            ] = round(
                adjusted_reranker_score,
                6,
            )

            candidate["retrieval_weight"] = round(
                retrieval_weight,
                6,
            )

            candidate["reranker_weight"] = round(
                reranker_weight,
                6,
            )

            candidate["reranker_gap"] = round(
                reranker_gap,
                6,
            )

            candidate[
                "candidate_reranker_gap"
            ] = round(
                reranker_gap,
                6,
            )

            candidate["final_score"] = round(
                _clamp_score(
                    final_score
                ),
                6,
            )

            candidate["reranked"] = True

        result["ranking_mode"] = (
            "retrieval_plus_qwen_reranker_dynamic"
        )

        result["reranker_applied"] = True

        result[
            "reranked_candidate_count"
        ] = actual_limit

    except Exception as exc:
        result["ranking_mode"] = (
            "retrieval_only_reranker_error"
        )

        result["ranking_error"] = (
            f"{type(exc).__name__}: {exc}"
        )

    _sort_and_update_decision(
        result
    )

    _apply_catboost_shadow(
        result=result,
        catboost_ranker_model=(
            catboost_ranker_model
        ),
    )

    return result


def _initialize_catboost_shadow_result(
    result: dict[str, Any],
) -> None:
    result["catboost_shadow_applied"] = False

    result[
        "catboost_shadow_candidate_count"
    ] = 0

    result["catboost_shadow_status"] = (
        "not_requested"
    )

    result["catboost_shadow_error"] = None

    result[
        "catboost_shadow_changed_winner"
    ] = False

    result[
        "catboost_shadow_best_company_id"
    ] = None

    result[
        "catboost_shadow_best_raw_score"
    ] = 0.0

    result[
        "catboost_shadow_best_normalized_score"
    ] = 0.0

    result[
        "catboost_shadow_raw_margin"
    ] = 0.0

    result[
        "catboost_shadow_normalized_margin"
    ] = 0.0


def _assign_retrieval_ranks(
    candidates: list[dict[str, Any]],
) -> None:
    """
    Reranker öncesindeki retrieval sırasını hesaplar.

    CatBoost eğitim veri setinde kullanılan retrieval_rank
    özelliğiyle aynı davranışı uygular.
    """

    ranked_indices = sorted(
        range(
            len(candidates)
        ),
        key=lambda index: (
            bool(
                candidates[index].get(
                    "unique_exact_match",
                    False,
                )
            ),
            _clamp_score(
                candidates[index].get(
                    "retrieval_score",
                    candidates[index].get(
                        "hybrid_score",
                        0.0,
                    ),
                )
            ),
            -index,
        ),
        reverse=True,
    )

    for retrieval_rank, candidate_index in enumerate(
        ranked_indices,
        start=1,
    ):
        candidates[
            candidate_index
        ]["retrieval_rank"] = (
            retrieval_rank
        )


def _assign_current_final_ranks(
    candidates: list[dict[str, Any]],
) -> None:
    for final_rank, candidate in enumerate(
        candidates,
        start=1,
    ):
        candidate[
            "current_final_rank"
        ] = final_rank


def _apply_catboost_shadow(
    *,
    result: dict[str, Any],
    catboost_ranker_model: (
        CatBoostRankerProvider | None
    ),
) -> None:
    """
    CatBoost alternatif sıralamasını hesaplar.

    Bu fonksiyon candidates listesini sıralamaz ve
    result["decision"] değerini değiştirmez.
    """

    candidates = result.get(
        "candidates",
        [],
    )

    _assign_current_final_ranks(
        candidates
    )

    if not candidates:
        result["catboost_shadow_status"] = (
            "skipped_no_candidates"
        )

        return

    if (
        catboost_ranker_model is None
        or not catboost_ranker_model.is_loaded
    ):
        result["catboost_shadow_status"] = (
            "model_not_loaded"
        )

        return

    if not result.get(
        "reranker_applied",
        False,
    ):
        result["catboost_shadow_status"] = (
            "skipped_reranker_not_applied"
        )

        return

    try:
        result_reranker_gap = _clamp_score(
            result.get(
                "reranker_gap",
                0.0,
            )
        )

        result_base_weight = _clamp_score(
            result.get(
                "base_reranker_weight",
                0.0,
            )
        )

        for candidate in candidates:
            candidate[
                "candidate_reranker_gap"
            ] = _clamp_score(
                candidate.get(
                    "candidate_reranker_gap",
                    candidate.get(
                        "reranker_gap",
                        result_reranker_gap,
                    ),
                )
            )

            candidate[
                "base_reranker_weight"
            ] = result_base_weight

        raw_scores = (
            catboost_ranker_model
            .predict_raw_scores(
                candidates
            )
        )

        if len(raw_scores) != len(
            candidates
        ):
            raise RuntimeError(
                "CatBoost skor sayısı aday "
                "sayısıyla eşleşmiyor."
            )

        cleaned_raw_scores = [
            _finite_float(
                score
            )
            for score in raw_scores
        ]

        normalized_scores = (
            _normalize_shadow_scores(
                cleaned_raw_scores
            )
        )

        ranked_indices = sorted(
            range(
                len(candidates)
            ),
            key=lambda index: (
                cleaned_raw_scores[index],
                -index,
            ),
            reverse=True,
        )

        catboost_rank_by_index = {
            candidate_index: rank
            for rank, candidate_index
            in enumerate(
                ranked_indices,
                start=1,
            )
        }

        for candidate_index, candidate in enumerate(
            candidates
        ):
            catboost_rank = (
                catboost_rank_by_index[
                    candidate_index
                ]
            )

            current_final_rank = int(
                candidate.get(
                    "current_final_rank",
                    candidate_index + 1,
                )
            )

            candidate[
                "catboost_raw_score"
            ] = round(
                cleaned_raw_scores[
                    candidate_index
                ],
                8,
            )

            candidate[
                "catboost_normalized_score"
            ] = round(
                normalized_scores[
                    candidate_index
                ],
                6,
            )

            candidate[
                "catboost_rank"
            ] = catboost_rank

            candidate[
                "catboost_shadow_top1"
            ] = (
                catboost_rank == 1
            )

            # Pozitif değer, CatBoost'un adayı mevcut
            # sıralamaya göre yukarı taşıdığını gösterir.
            candidate[
                "catboost_shadow_order_delta"
            ] = (
                current_final_rank
                - catboost_rank
            )

        best_shadow_index = (
            ranked_indices[0]
        )

        best_shadow_candidate = candidates[
            best_shadow_index
        ]

        current_best_candidate = (
            candidates[0]
        )

        best_raw_score = (
            cleaned_raw_scores[
                best_shadow_index
            ]
        )

        best_normalized_score = (
            normalized_scores[
                best_shadow_index
            ]
        )

        if len(ranked_indices) > 1:
            second_shadow_index = (
                ranked_indices[1]
            )

            second_raw_score = (
                cleaned_raw_scores[
                    second_shadow_index
                ]
            )

            second_normalized_score = (
                normalized_scores[
                    second_shadow_index
                ]
            )

        else:
            second_raw_score = 0.0
            second_normalized_score = 0.0

        current_best_company_id = (
            current_best_candidate.get(
                "company_id"
            )
        )

        shadow_best_company_id = (
            best_shadow_candidate.get(
                "company_id"
            )
        )

        result["catboost_shadow_applied"] = True

        result[
            "catboost_shadow_candidate_count"
        ] = len(
            candidates
        )

        result["catboost_shadow_status"] = (
            "applied"
        )

        result["catboost_shadow_error"] = None

        result[
            "catboost_shadow_changed_winner"
        ] = (
            shadow_best_company_id
            != current_best_company_id
        )

        result[
            "catboost_shadow_best_company_id"
        ] = shadow_best_company_id

        result[
            "catboost_shadow_best_raw_score"
        ] = round(
            best_raw_score,
            8,
        )

        result[
            "catboost_shadow_best_normalized_score"
        ] = round(
            best_normalized_score,
            6,
        )

        result[
            "catboost_shadow_raw_margin"
        ] = round(
            best_raw_score
            - second_raw_score,
            8,
        )

        result[
            "catboost_shadow_normalized_margin"
        ] = round(
            max(
                0.0,
                best_normalized_score
                - second_normalized_score,
            ),
            6,
        )

    except Exception as exc:
        result["catboost_shadow_applied"] = False

        result[
            "catboost_shadow_candidate_count"
        ] = 0

        result["catboost_shadow_status"] = (
            "error"
        )

        result["catboost_shadow_error"] = (
            f"{type(exc).__name__}: {exc}"
        )


def _normalize_shadow_scores(
    raw_scores: Sequence[float],
) -> list[float]:
    """
    Aynı sorgunun CatBoost skorlarını min-max yöntemiyle
    0-1 aralığına dönüştürür.

    Bu değer olasılık değildir.
    """

    if not raw_scores:
        return []

    cleaned_scores = [
        _finite_float(
            score
        )
        for score in raw_scores
    ]

    if len(cleaned_scores) == 1:
        return [
            1.0
        ]

    minimum_score = min(
        cleaned_scores
    )

    maximum_score = max(
        cleaned_scores
    )

    score_range = (
        maximum_score
        - minimum_score
    )

    if score_range <= 1e-12:
        return [
            0.5
            for _ in cleaned_scores
        ]

    return [
        max(
            0.0,
            min(
                1.0,
                (
                    score
                    - minimum_score
                )
                / score_range,
            ),
        )
        for score in cleaned_scores
    ]


def _finite_float(
    value: Any,
) -> float:
    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

    if not math.isfinite(
        numeric_value
    ):
        return 0.0

    return numeric_value


def _calculate_query_length(
    query_text: str,
) -> int:
    """
    Boşluk ve noktalama işaretleri çıkarıldıktan sonraki
    alfanümerik karakter sayısını döndürür.
    """

    return sum(
        character.isalnum()
        for character in query_text
    )


def _clamp_score(
    value: Any,
) -> float:
    numeric_value = _finite_float(
        value
    )

    return max(
        0.0,
        min(
            1.0,
            numeric_value,
        ),
    )


def _calculate_lexical_evidence(
    candidate: dict[str, Any],
) -> float:
    """
    Adayın isim temelli en güçlü lexical kanıtını hesaplar.

    Embedding ve reranker skorları burada kullanılmaz.
    """

    lexical_scores = (
        candidate.get(
            "char_wb_tfidf_score",
            0.0,
        ),
        candidate.get(
            "raw_char_tfidf_score",
            0.0,
        ),
        candidate.get(
            "char_tfidf_score",
            0.0,
        ),
        candidate.get(
            "fuzzy_ratio_score",
            0.0,
        ),
        candidate.get(
            "fuzzy_wratio_score",
            0.0,
        ),
        candidate.get(
            "jaro_winkler_score",
            0.0,
        ),
        candidate.get(
            "token_prefix_score",
            0.0,
        ),
    )

    return max(
        _clamp_score(score)
        for score in lexical_scores
    )


def _calculate_reranker_gap(
    reranker_scores: Sequence[float],
) -> float:
    if not reranker_scores:
        return 0.0

    if len(reranker_scores) == 1:
        return 1.0

    sorted_scores = sorted(
        (
            _clamp_score(score)
            for score in reranker_scores
        ),
        reverse=True,
    )

    return max(
        0.0,
        sorted_scores[0]
        - sorted_scores[1],
    )


def _determine_base_reranker_weight(
    *,
    query_length: int,
    query_token_count: int,
    reranker_gap: float,
) -> float:
    if query_length <= 5:
        weight = 0.15

    elif (
        query_token_count == 1
        and query_length <= 10
    ):
        weight = 0.20

    elif query_token_count == 1:
        weight = 0.25

    elif query_length <= 12:
        weight = 0.30

    else:
        weight = 0.40

    if reranker_gap < 0.03:
        weight = min(
            weight,
            0.15,
        )

    elif reranker_gap < 0.06:
        weight = min(
            weight,
            0.25,
        )

    return weight


def _adjust_reranker_score(
    *,
    reranker_score: float,
    lexical_evidence: float,
) -> float:
    reranker_score = _clamp_score(
        reranker_score
    )

    lexical_evidence = _clamp_score(
        lexical_evidence
    )

    if lexical_evidence < 0.20:
        return min(
            reranker_score,
            0.35,
        )

    if lexical_evidence < 0.30:
        return min(
            reranker_score,
            0.50,
        )

    if lexical_evidence < 0.40:
        return min(
            reranker_score,
            0.65,
        )

    return reranker_score


def _determine_candidate_reranker_weight(
    *,
    candidate: dict[str, Any],
    base_reranker_weight: float,
    lexical_evidence: float,
) -> float:
    reranker_weight = _clamp_score(
        base_reranker_weight
    )

    lexical_evidence = _clamp_score(
        lexical_evidence
    )

    if lexical_evidence < 0.20:
        reranker_weight = 0.0

    elif lexical_evidence < 0.30:
        reranker_weight = min(
            reranker_weight,
            0.10,
        )

    elif lexical_evidence < 0.40:
        reranker_weight = min(
            reranker_weight,
            0.20,
        )

    if candidate.get(
        "identifier_overlap",
        False,
    ):
        reranker_weight = min(
            reranker_weight,
            0.10,
        )

    if candidate.get(
        "ambiguous_exact_match",
        False,
    ):
        reranker_weight = min(
            reranker_weight,
            0.20,
        )

    return max(
        0.0,
        min(
            0.50,
            reranker_weight,
        ),
    )


def _build_reranker_query(
    result: dict[str, Any],
) -> str:
    normalization = result[
        "normalization"
    ]

    raw_text = str(
        normalization.get(
            "raw_text",
            "",
        )
    ).strip()

    normalized_text = str(
        result.get(
            "query_text",
            "",
        )
    ).strip()

    query_parts: list[str] = []

    if raw_text:
        query_parts.append(
            "Observed entity text: "
            f"{raw_text}"
        )

    if (
        normalized_text
        and normalized_text.casefold()
        != raw_text.casefold()
    ):
        query_parts.append(
            "Normalized entity text: "
            f"{normalized_text}"
        )

    if not query_parts:
        raise ValueError(
            "Reranker sorgu metni üretilemedi."
        )

    return " | ".join(
        query_parts
    )


def _build_candidate_document(
    candidate: dict[str, Any],
) -> str:
    legal_name = str(
        candidate.get(
            "legal_name",
            "",
        )
    ).strip()

    brand_name = str(
        candidate.get(
            "brand_name",
            "",
        )
    ).strip()

    matched_alias = str(
        candidate.get(
            "matched_alias",
            "",
        )
    ).strip()

    normalized_alias = str(
        candidate.get(
            "normalized_alias",
            "",
        )
    ).strip()

    alias_type = str(
        candidate.get(
            "alias_type",
            "",
        )
    ).strip()

    identifier_overlap = bool(
        candidate.get(
            "identifier_overlap",
            False,
        )
    )

    identifier_score = _clamp_score(
        candidate.get(
            "identifier_score",
            0.0,
        )
    )

    document_parts: list[str] = []

    if legal_name:
        document_parts.append(
            "Candidate legal name: "
            f"{legal_name}"
        )

    if (
        brand_name
        and brand_name.casefold()
        != legal_name.casefold()
    ):
        document_parts.append(
            "Candidate brand name: "
            f"{brand_name}"
        )

    if (
        matched_alias
        and matched_alias.casefold()
        not in {
            legal_name.casefold(),
            brand_name.casefold(),
        }
    ):
        document_parts.append(
            "Candidate known alias: "
            f"{matched_alias}"
        )

    if normalized_alias:
        document_parts.append(
            "Candidate normalized alias: "
            f"{normalized_alias}"
        )

    if alias_type:
        document_parts.append(
            "Alias type: "
            f"{alias_type}"
        )

    document_parts.append(
        "Identifier overlap: "
        + (
            "yes"
            if identifier_overlap
            else "no"
        )
    )

    document_parts.append(
        "Identifier match score: "
        f"{identifier_score:.3f}"
    )

    if not document_parts:
        raise ValueError(
            "Reranker aday belgesi üretilemedi."
        )

    return " | ".join(
        document_parts
    )


def _sort_and_update_decision(
    result: dict[str, Any],
) -> None:
    candidates = result[
        "candidates"
    ]

    if not candidates:
        result["best_candidate"] = None
        result["score_margin"] = 0.0
        result["decision"] = "UNKNOWN"

        return

    candidates.sort(
        key=lambda candidate: (
            bool(
                candidate.get(
                    "unique_exact_match",
                    False,
                )
            ),
            float(
                candidate[
                    "final_score"
                ]
            ),
        ),
        reverse=True,
    )

    _assign_current_final_ranks(
        candidates
    )

    best_candidate = candidates[0]

    best_score = float(
        best_candidate[
            "final_score"
        ]
    )

    second_score = (
        float(
            candidates[1][
                "final_score"
            ]
        )
        if len(candidates) > 1
        else 0.0
    )

    margin = round(
        best_score
        - second_score,
        6,
    )

    result["best_candidate"] = (
        best_candidate
    )

    result["score_margin"] = (
        margin
    )

    result["decision"] = (
        _make_decision(
            best_candidate=(
                best_candidate
            ),
            margin=margin,
        )
    )


def _make_decision(
    best_candidate: dict[str, Any],
    margin: float,
) -> str:
    if best_candidate.get(
        "unique_exact_match",
        False,
    ):
        return "AUTO_MATCH"

    if best_candidate.get(
        "ambiguous_exact_match",
        False,
    ):
        return "MANUAL_REVIEW"

    final_score = float(
        best_candidate[
            "final_score"
        ]
    )

    retrieval_score = float(
        best_candidate[
            "retrieval_score"
        ]
    )

    adjusted_reranker_score = float(
        best_candidate.get(
            "adjusted_reranker_score",
            best_candidate.get(
                "reranker_score",
                0.0,
            ),
        )
    )

    lexical_evidence = float(
        best_candidate.get(
            "lexical_evidence",
            0.0,
        )
    )

    if (
        final_score >= 0.90
        and retrieval_score >= 0.70
        and adjusted_reranker_score >= 0.70
        and lexical_evidence >= 0.55
        and margin >= 0.08
    ):
        return "AUTO_MATCH"

    if final_score >= 0.58:
        return "MANUAL_REVIEW"

    return "UNKNOWN"