from __future__ import annotations

from typing import Sequence

from app.ranking.company_reranking import (
    apply_company_reranking,
)


class FakeReranker:
    is_loaded = True

    def score(
        self,
        query: str,
        documents: Sequence[str],
    ) -> list[float]:
        assert query
        assert len(documents) == 2

        return [
            0.20,
            0.95,
        ]


def build_candidate(
    company_id: int,
    legal_name: str,
    hybrid_score: float,
) -> dict:
    return {
        "company_id": company_id,
        "legal_name": legal_name,
        "brand_name": legal_name,
        "city": "İstanbul",
        "sector": "Test",
        "matched_alias": legal_name,
        "normalized_alias": (
            legal_name.lower()
        ),
        "alias_type": "TEST",
        "exact_match": False,
        "unique_exact_match": False,
        "ambiguous_exact_match": False,
        "identifier_score": 0.0,
        "identifier_overlap": False,
        "char_tfidf_score": (
            hybrid_score
        ),
        "fuzzy_ratio_score": (
            hybrid_score
        ),
        "fuzzy_partial_ratio_score": (
            hybrid_score
        ),
        "fuzzy_token_set_score": (
            hybrid_score
        ),
        "fuzzy_wratio_score": (
            hybrid_score
        ),
        "embedding_score": (
            hybrid_score
        ),
        "hybrid_score": (
            hybrid_score
        ),
    }


def test_reranker_can_change_candidate_order() -> None:
    result = {
        "normalization": {
            "raw_text": "TEST EFT",
        },
        "query_text": "test",
        "decision": "MANUAL_REVIEW",
        "best_candidate": None,
        "score_margin": 0.0,
        "candidates": [
            build_candidate(
                company_id=1,
                legal_name=(
                    "Birinci Şirket"
                ),
                hybrid_score=0.80,
            ),
            build_candidate(
                company_id=2,
                legal_name=(
                    "İkinci Şirket"
                ),
                hybrid_score=0.76,
            ),
        ],
        "index_stats": {},
    }

    reranked_result = (
        apply_company_reranking(
            result=result,
            reranker_model=(
                FakeReranker()
            ),
            rerank_limit=2,
        )
    )

    assert (
        reranked_result[
            "reranker_applied"
        ]
        is True
    )

    assert (
        reranked_result[
            "best_candidate"
        ][
            "company_id"
        ]
        == 2
    )

    assert (
        reranked_result[
            "candidates"
        ][0][
            "reranked"
        ]
        is True
    )