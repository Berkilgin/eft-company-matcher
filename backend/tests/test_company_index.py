from __future__ import annotations

from pathlib import Path

import pytest

from app.retrieval.company_index import CompanyIndex


@pytest.fixture
def company_index(
    tmp_path: Path,
) -> CompanyIndex:
    companies_path = tmp_path / "companies.csv"
    aliases_path = tmp_path / "aliases.csv"

    companies_path.write_text(
        (
            "company_id,legal_name,brand_name,"
            "city,sector,tax_number\n"
            "1001,Türkiye Petrol Rafinerileri A.Ş.,"
            "TÜPRAŞ,Kocaeli,Petrol,1111111111\n"
            "1002,Türkiye Petrolleri Anonim Ortaklığı,"
            "TPAO,Ankara,Enerji,2222222222\n"
            "1003,Petrol Ofisi A.Ş.,"
            "Petrol Ofisi,İstanbul,Akaryakıt,3333333333\n"
        ),
        encoding="utf-8",
    )

    aliases_path.write_text(
        (
            "alias_id,company_id,alias_text,alias_type\n"
            "1,1001,TUPRAS,BRAND\n"
            "2,1001,TR PETROL RAF,ABBREVIATION\n"
            "3,1001,TUPRAS IZMIT,OBSERVED_EFT\n"
            "4,1002,TPAO,BRAND\n"
            "5,1002,TURKIYE PETROLLERI,SHORT_NAME\n"
            "6,1003,PETROL OFISI,BRAND\n"
            "7,1003,POAS,ABBREVIATION\n"
        ),
        encoding="utf-8",
    )

    return CompanyIndex.from_csv(
        companies_path=companies_path,
        aliases_path=aliases_path,
    )


def test_exact_alias_match(
    company_index: CompanyIndex,
) -> None:
    result = company_index.search(
        "TUPRAS",
        limit=3,
    )

    assert result["decision"] == "AUTO_MATCH"
    assert result["best_candidate"]["company_id"] == 1001
    assert result["best_candidate"]["exact_match"] is True


def test_typo_match(
    company_index: CompanyIndex,
) -> None:
    result = company_index.search(
        "TUPRASS FATURA ODEMESI",
        limit=3,
    )

    assert result["best_candidate"]["company_id"] == 1001
    assert result["best_candidate"]["hybrid_score"] > 0.70


def test_abbreviation_match(
    company_index: CompanyIndex,
) -> None:
    result = company_index.search(
        "TR PET RAF IZM ODM",
        limit=3,
    )

    assert result["best_candidate"]["company_id"] == 1001


def test_petrol_ofisi_match(
    company_index: CompanyIndex,
) -> None:
    result = company_index.search(
        "PETROL OFS AS FATURA",
        limit=3,
    )

    assert result["best_candidate"]["company_id"] == 1003


def test_index_stats(
    company_index: CompanyIndex,
) -> None:
    stats = company_index.get_stats()

    assert stats["company_count"] == 3
    assert stats["alias_count"] >= 7
    assert stats["tfidf_feature_count"] > 0