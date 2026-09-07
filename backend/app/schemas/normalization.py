from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class NormalizationRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
        description="Normalize edilecek EFT açıklaması",
        examples=[
            "TÜPRAŞ A.Ş. İZMİT RAF. FATURA ÖDM. REF:398192"
        ],
    )

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError("EFT açıklaması boş bırakılamaz.")

        return cleaned_value


class ExtractedIdentifiers(BaseModel):
    ibans: list[str] = Field(default_factory=list)
    tax_numbers: list[str] = Field(default_factory=list)
    identity_numbers: list[str] = Field(default_factory=list)


class NormalizationResponse(BaseModel):
    raw_text: str
    normalized_text: str
    ascii_text: str
    core_text: str

    tokens: list[str]
    legal_types: list[str]
    removed_tokens: list[str]
    expanded_tokens: dict[str, str]

    identifiers: ExtractedIdentifiers
    reference_numbers: list[str]