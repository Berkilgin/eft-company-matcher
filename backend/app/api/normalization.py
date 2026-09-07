from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.normalization.normalizer import eft_normalizer
from app.schemas.normalization import (
    NormalizationRequest,
    NormalizationResponse,
)


router = APIRouter(
    prefix="/normalization",
    tags=["Normalization"],
)


@router.post(
    "/normalize",
    response_model=NormalizationResponse,
    status_code=status.HTTP_200_OK,
    summary="EFT açıklamasını normalize et",
)
def normalize_eft_description(
    request: NormalizationRequest,
) -> NormalizationResponse:
    try:
        result = eft_normalizer.normalize(request.text)

    except (TypeError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return NormalizationResponse.model_validate(result)


@router.get(
    "/example",
    response_model=NormalizationResponse,
    summary="Örnek EFT açıklamasını normalize et",
)
def normalize_example() -> NormalizationResponse:
    example_text = (
        "TÜPRAŞ A.Ş. İZMİT RAF. "
        "FATURA ÖDM. REF:398192"
    )

    result = eft_normalizer.normalize(example_text)
    return NormalizationResponse.model_validate(result)