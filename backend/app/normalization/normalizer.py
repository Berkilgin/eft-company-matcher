from __future__ import annotations

import re
import unicodedata
from typing import Any

from app.normalization.dictionaries import (
    ABBREVIATIONS,
    BANKING_STOPWORDS,
    LEGAL_SUFFIX_PATTERNS,
)


# Türkçede Python lower işleminin yerel ayardan bağımsız ve
# öngörülebilir olması için I/İ karakterlerini önceden dönüştürüyoruz.
TURKISH_LOWER_TRANSLATION = str.maketrans(
    {
        "I": "ı",
        "İ": "i",
    }
)


# Şirket isimlerini aramada Türkçe ve ASCII biçimleri birlikte tutulacak.
TURKISH_ASCII_TRANSLATION = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "I": "I",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)


WHITESPACE_PATTERN = re.compile(r"\s+")
NON_ALPHANUMERIC_PATTERN = re.compile(r"[^a-z0-9]+")
UNICODE_PUNCTUATION_PATTERN = re.compile(r"[^\w\s]+", re.UNICODE)

DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}"
    r"|"
    r"\d{4}[./-]\d{1,2}[./-]\d{1,2}"
    r")\b"
)

TIME_PATTERN = re.compile(
    r"\b\d{1,2}:\d{2}(?::\d{2})?\b"
)

# Referans etiketiyle birlikte gelen değerleri kaldırır.
REFERENCE_PATTERN = re.compile(
    r"\b(?:"
    r"ref(?:erans)?"
    r"|dekont"
    r"|islem\s*no"
    r"|txn"
    r"|trx"
    r"|fatura\s*no"
    r")"
    r"\s*[:#=-]?\s*"
    r"([a-z0-9-]{3,})\b",
    re.IGNORECASE,
)

# Açıklama içindeki 6 veya daha uzun saf sayıları işlem numarası
# olma ihtimali yüksek olduğu için çekirdek metinden çıkarıyoruz.
LONG_NUMBER_PATTERN = re.compile(r"\b\d{6,}\b")

IBAN_PATTERN = re.compile(
    r"\bTR(?:\s*\d){24}\b",
    re.IGNORECASE,
)

TAX_NUMBER_PATTERN = re.compile(
    r"\b(?:"
    r"vkn"
    r"|vergi\s*(?:kimlik\s*)?(?:no|numarasi)?"
    r")"
    r"\s*[:#=-]?\s*"
    r"(\d{10})\b",
    re.IGNORECASE,
)

IDENTITY_NUMBER_PATTERN = re.compile(
    r"\b(?:"
    r"tckn"
    r"|tc\s*kimlik\s*(?:no|numarasi)?"
    r")"
    r"\s*[:#=-]?\s*"
    r"(\d{11})\b",
    re.IGNORECASE,
)


class EFTNormalizer:
    """
    EFT açıklamalarını şirket eşleştirme sistemine uygun biçimde normalize eder.

    Bu sınıf herhangi bir yapay zekâ modeli kullanmaz. Deterministik çalışır.
    Aynı girdi her zaman aynı çıktıyı üretir.
    """

    def normalize(self, raw_text: str) -> dict[str, Any]:
        if not isinstance(raw_text, str):
            raise TypeError("raw_text bir metin olmalıdır.")

        raw_text = raw_text.strip()

        if not raw_text:
            raise ValueError("EFT açıklaması boş olamaz.")

        unicode_text = unicodedata.normalize("NFKC", raw_text)
        lowercase_text = self._turkish_lower(unicode_text)
        ascii_source_text = self._to_ascii(lowercase_text)

        identifiers = self._extract_identifiers(unicode_text)
        reference_numbers = self._extract_reference_numbers(
            ascii_source_text
        )

        normalized_text = self._normalize_unicode_surface(lowercase_text)
        ascii_text = self._normalize_ascii_surface(ascii_source_text)

        working_text = ascii_source_text

        # Kimlik bilgilerini çekirdek şirket metnine dahil etmiyoruz.
        working_text = IBAN_PATTERN.sub(" ", working_text)
        working_text = TAX_NUMBER_PATTERN.sub(" ", working_text)
        working_text = IDENTITY_NUMBER_PATTERN.sub(" ", working_text)

        # İşlem referansları, tarihler ve saatler kaldırılır.
        working_text = REFERENCE_PATTERN.sub(" ", working_text)
        working_text = DATE_PATTERN.sub(" ", working_text)
        working_text = TIME_PATTERN.sub(" ", working_text)

        legal_types, working_text = self._extract_and_remove_legal_types(
            working_text
        )

        working_text = LONG_NUMBER_PATTERN.sub(" ", working_text)
        working_text = NON_ALPHANUMERIC_PATTERN.sub(" ", working_text)
        working_text = self._collapse_spaces(working_text)

        original_tokens = working_text.split()

        core_tokens: list[str] = []
        removed_tokens: list[str] = []
        expanded_tokens: dict[str, str] = {}

        for token in original_tokens:
            if token in BANKING_STOPWORDS:
                removed_tokens.append(token)
                continue

            expanded_token = ABBREVIATIONS.get(token, token)

            if expanded_token != token:
                expanded_tokens[token] = expanded_token

            if expanded_token in BANKING_STOPWORDS:
                removed_tokens.append(token)
                continue

            core_tokens.append(expanded_token)

        core_tokens = self._remove_consecutive_duplicates(core_tokens)
        removed_tokens = self._unique_preserving_order(removed_tokens)

        return {
            "raw_text": raw_text,
            "normalized_text": normalized_text,
            "ascii_text": ascii_text,
            "core_text": " ".join(core_tokens),
            "tokens": core_tokens,
            "legal_types": legal_types,
            "removed_tokens": removed_tokens,
            "expanded_tokens": expanded_tokens,
            "identifiers": identifiers,
            "reference_numbers": reference_numbers,
        }

    @staticmethod
    def _turkish_lower(text: str) -> str:
        return text.translate(TURKISH_LOWER_TRANSLATION).lower()

    @staticmethod
    def _to_ascii(text: str) -> str:
        return text.translate(TURKISH_ASCII_TRANSLATION)

    @staticmethod
    def _collapse_spaces(text: str) -> str:
        return WHITESPACE_PATTERN.sub(" ", text).strip()

    def _normalize_unicode_surface(self, text: str) -> str:
        cleaned = UNICODE_PUNCTUATION_PATTERN.sub(" ", text)
        return self._collapse_spaces(cleaned)

    def _normalize_ascii_surface(self, text: str) -> str:
        cleaned = NON_ALPHANUMERIC_PATTERN.sub(" ", text)
        return self._collapse_spaces(cleaned)

    def _extract_and_remove_legal_types(
        self,
        text: str,
    ) -> tuple[list[str], str]:
        detected_types: list[str] = []
        working_text = text

        for canonical_name, pattern_text in LEGAL_SUFFIX_PATTERNS:
            pattern = re.compile(pattern_text, re.IGNORECASE)

            if pattern.search(working_text):
                detected_types.append(canonical_name)
                working_text = pattern.sub(" ", working_text)

        return (
            self._unique_preserving_order(detected_types),
            working_text,
        )

    def _extract_identifiers(
        self,
        text: str,
    ) -> dict[str, list[str]]:
        ascii_upper_text = self._to_ascii(text).upper()

        ibans: list[str] = []

        for match in IBAN_PATTERN.finditer(ascii_upper_text):
            normalized_iban = re.sub(
                r"[^A-Z0-9]",
                "",
                match.group(0),
            )

            if normalized_iban not in ibans:
                ibans.append(normalized_iban)

        tax_numbers = self._unique_preserving_order(
            TAX_NUMBER_PATTERN.findall(ascii_upper_text)
        )

        identity_numbers = self._unique_preserving_order(
            IDENTITY_NUMBER_PATTERN.findall(ascii_upper_text)
        )

        return {
            "ibans": ibans,
            "tax_numbers": tax_numbers,
            "identity_numbers": identity_numbers,
        }

    def _extract_reference_numbers(
        self,
        ascii_text: str,
    ) -> list[str]:
        return self._unique_preserving_order(
            REFERENCE_PATTERN.findall(ascii_text)
        )

    @staticmethod
    def _remove_consecutive_duplicates(
        values: list[str],
    ) -> list[str]:
        if not values:
            return []

        result = [values[0]]

        for value in values[1:]:
            if value != result[-1]:
                result.append(value)

        return result

    @staticmethod
    def _unique_preserving_order(
        values: list[str],
    ) -> list[str]:
        return list(dict.fromkeys(values))


eft_normalizer = EFTNormalizer()