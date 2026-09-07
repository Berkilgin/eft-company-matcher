from __future__ import annotations

from typing import Any, Protocol, Sequence


class RerankerProvider(Protocol):
    is_loaded: bool

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        ...


def apply_company_reranking(
    result: dict[str, Any],
    reranker_model: RerankerProvider | None,
    rerank_limit: int = 10,
) -> dict[str, Any]:
    """
    Retrieval adaylarını Qwen3 Reranker ile yeniden sıralar.

    Reranker ağırlığı sorgu uzunluğuna, lexical kanıta,
    identifier eşleşmesine ve reranker skor farkına göre
    dinamik olarak belirlenir.
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

        candidate[
            "retrieval_score"
        ] = round(
            retrieval_score,
            6,
        )

        candidate[
            "reranker_score"
        ] = 0.0

        candidate[
            "adjusted_reranker_score"
        ] = 0.0

        candidate[
            "lexical_evidence"
        ] = round(
            lexical_evidence,
            6,
        )

        candidate[
            "retrieval_weight"
        ] = 1.0

        candidate[
            "reranker_weight"
        ] = 0.0

        candidate[
            "reranker_gap"
        ] = 0.0

        candidate[
            "query_length"
        ] = query_length

        candidate[
            "query_token_count"
        ] = query_token_count

        candidate[
            "final_score"
        ] = round(
            retrieval_score,
            6,
        )

        candidate[
            "reranked"
        ] = False

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

    if not candidates:
        result["best_candidate"] = None
        result["decision"] = "UNKNOWN"
        result["score_margin"] = 0.0

        return result

    if (
        reranker_model is None
        or not reranker_model.is_loaded
    ):
        _sort_and_update_decision(
            result
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
        for candidate
        in candidates_to_rerank
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
            for score
            in raw_reranker_scores
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

            candidate[
                "reranker_score"
            ] = round(
                reranker_score,
                6,
            )

            candidate[
                "adjusted_reranker_score"
            ] = round(
                adjusted_reranker_score,
                6,
            )

            candidate[
                "retrieval_weight"
            ] = round(
                retrieval_weight,
                6,
            )

            candidate[
                "reranker_weight"
            ] = round(
                reranker_weight,
                6,
            )

            candidate[
                "reranker_gap"
            ] = round(
                reranker_gap,
                6,
            )

            candidate[
                "final_score"
            ] = round(
                _clamp_score(
                    final_score
                ),
                6,
            )

            candidate[
                "reranked"
            ] = True

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

    return result


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
    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return 0.0

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

    Embedding ve reranker skorları burada bilinçli olarak
    kullanılmaz.
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
    """
    En yüksek iki reranker skoru arasındaki farkı hesaplar.

    Tek aday varsa reranker kararsızlığı bulunmadığı kabul
    edilerek 1.0 döndürülür.
    """

    if not reranker_scores:
        return 0.0

    if len(reranker_scores) == 1:
        return 1.0

    sorted_scores = sorted(
        (
            _clamp_score(score)
            for score
            in reranker_scores
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
    """
    Sorgu yapısına göre temel reranker ağırlığını belirler.

    Çok kısa ve tek tokenlı sorgularda Qwen skorunun
    sıralamayı aşırı etkilemesi engellenir.
    """

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

    # Reranker ilk iki aday arasında anlamlı ayrım
    # yapamıyorsa etkisi azaltılır.
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
    """
    Lexical kanıt zayıfken aşırı yüksek model skorlarını
    sınırlar.

    Örneğin alakasız bir şirket yalnızca semantik veya
    popülerlik yanlılığı nedeniyle 0.95 skor aldıysa,
    nihai sıralamayı tek başına değiştiremez.
    """

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
    """
    Aday özelinde reranker ağırlığını belirler.
    """

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

    # Uzun sayısal veya alfanümerik identifier eşleşmesi
    # güçlü ve deterministik bir retrieval kanıtıdır.
    if candidate.get(
        "identifier_overlap",
        False,
    ):
        reranker_weight = min(
            reranker_weight,
            0.10,
        )

    # Aynı normalize alias birden fazla şirkette varsa
    # reranker sıralamada yardımcı olabilir; ancak ağırlığı
    # sınırlandırılır ve karar MANUAL_REVIEW olarak kalır.
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
    """
    Reranker sorgusunu bankacılık bağlamından arındırır.
    """

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
    """
    Reranker belgesini yalnızca şirket adı kanıtlarından
    oluşturur.
    """

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