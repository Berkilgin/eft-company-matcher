from __future__ import annotations

import pytest

from app.normalization.normalizer import EFTNormalizer


@pytest.fixture
def normalizer() -> EFTNormalizer:
    return EFTNormalizer()


def test_tupras_example(normalizer: EFTNormalizer) -> None:
    result = normalizer.normalize(
        "TÜPRAŞ A.Ş. İZMİT RAF. FATURA ÖDM. REF:398192"
    )

    assert result["core_text"] == "tupras izmit rafineri"
    assert result["legal_types"] == ["ANONIM_SIRKETI"]
    assert "fatura" in result["removed_tokens"]
    assert "odm" in result["removed_tokens"]
    assert result["reference_numbers"] == ["398192"]


def test_turkish_ascii_conversion(normalizer: EFTNormalizer) -> None:
    result = normalizer.normalize(
        "ŞİŞECAM SANAYİ A.Ş. HAVALE"
    )

    assert "sisecam" in result["core_text"]
    assert "sanayi" in result["core_text"]
    assert "havale" not in result["core_text"]


def test_limited_company_detection(normalizer: EFTNormalizer) -> None:
    result = normalizer.normalize(
        "AKDENİZ GIDA LTD. ŞTİ. FATURA BEDELİ"
    )

    assert result["core_text"] == "akdeniz gida"
    assert result["legal_types"] == ["LIMITED_SIRKETI"]


def test_iban_extraction(normalizer: EFTNormalizer) -> None:
    result = normalizer.normalize(
        "TUPRAS ODEME TR12 3456 7890 1234 5678 9012 34"
    )

    assert result["identifiers"]["ibans"] == [
        "TR123456789012345678901234"
    ]

    assert "tr12" not in result["core_text"]


def test_tax_number_extraction(normalizer: EFTNormalizer) -> None:
    result = normalizer.normalize(
        "ABC MUH LTD STI VKN:1234567890 FATURA"
    )

    assert result["identifiers"]["tax_numbers"] == [
        "1234567890"
    ]

    assert result["core_text"] == "abc muhendislik"


def test_empty_text_raises_error(
    normalizer: EFTNormalizer,
) -> None:
    with pytest.raises(ValueError):
        normalizer.normalize("   ")