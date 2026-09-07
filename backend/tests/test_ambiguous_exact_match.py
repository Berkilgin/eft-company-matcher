from __future__ import annotations

from app.ranking.company_reranking import (
    apply_company_reranking,
)
from app.retrieval.company_index import (
    AliasRecord,
    CompanyIndex,
    CompanyRecord,
)


class IdentifierAwareFakeReranker:
    is_loaded = True

    def score(
        self,
        query: str,
        documents: list[str],
    ) -> list[float]:
        return [
            0.95
            if "123456789" in document
            else 0.20
            for document in documents
        ]


def build_index() -> CompanyIndex:
    companies = {
        1: CompanyRecord(
            company_id=1,
            legal_name=(
                "Custody Bank "
                "of Japan /123456789"
            ),
            brand_name=(
                "Custody Bank "
                "of Japan"
            ),
            city="Tokyo",
            sector="Bank",
            tax_number="",
        ),
        2: CompanyRecord(
            company_id=2,
            legal_name=(
                "Custody Bank "
                "of Japan /999999999"
            ),
            brand_name=(
                "Custody Bank "
                "of Japan"
            ),
            city="Tokyo",
            sector="Bank",
            tax_number="",
        ),
    }

    aliases = [
        AliasRecord(
            alias_id=1,
            company_id=1,
            alias_text=(
                "Custody Bank "
                "of Japan /123456789"
            ),
            normalized_alias=(
                "custody bank of japan"
            ),
            alias_type="LEGAL_NAME",
        ),
        AliasRecord(
            alias_id=2,
            company_id=2,
            alias_text=(
                "Custody Bank "
                "of Japan /999999999"
            ),
            normalized_alias=(
                "custody bank of japan"
            ),
            alias_type="LEGAL_NAME",
        ),
    ]

    return CompanyIndex(
        companies=companies,
        aliases=aliases,
    )


def test_duplicate_exact_alias_is_ambiguous() -> None:
    index = build_index()

    result = index.search(
        text=(
            "CUSTODY BANK OF JAPAN "
            "123456789 ODEME"
        ),
        limit=2,
    )

    best_candidate = result[
        "best_candidate"
    ]

    assert (
        best_candidate[
            "company_id"
        ]
        == 1
    )

    assert (
        best_candidate[
            "exact_match"
        ]
        is True
    )

    assert (
        best_candidate[
            "unique_exact_match"
        ]
        is False
    )

    assert (
        best_candidate[
            "ambiguous_exact_match"
        ]
        is True
    )

    assert (
        best_candidate[
            "identifier_overlap"
        ]
        is True
    )

    assert (
        result["decision"]
        == "MANUAL_REVIEW"
    )


def test_ambiguous_exact_does_not_become_auto_match() -> None:
    index = build_index()

    result = index.search(
        text=(
            "CUSTODY BANK OF JAPAN "
            "123456789 ODEME"
        ),
        limit=2,
    )

    reranked_result = (
        apply_company_reranking(
            result=result,
            reranker_model=(
                IdentifierAwareFakeReranker()
            ),
            rerank_limit=2,
        )
    )

    assert (
        reranked_result[
            "best_candidate"
        ][
            "company_id"
        ]
        == 1
    )

    assert (
        reranked_result[
            "decision"
        ]
        == "MANUAL_REVIEW"
    )