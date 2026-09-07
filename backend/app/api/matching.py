from __future__ import annotations

import logging
from typing import Any

from fastapi import (
    APIRouter,
    HTTPException,
    status,
)

from app.models.catboost_ranker_model import (
    CatBoostCompanyRanker,
)
from app.ranking.company_reranking import (
    apply_company_reranking,
)
from app.schemas.matching import (
    CompanyMatchRequest,
    CompanyMatchResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/matching",
    tags=["Company Matching"],
)


_company_index: Any | None = None
_reranker_model: Any | None = None


_catboost_shadow_model = CatBoostCompanyRanker()
_catboost_shadow_load_error: str | None = None


try:
    _catboost_shadow_model.load()

    logger.info(
        "CatBoostRanker shadow modeli yüklendi. "
        "Feature count: %s",
        _catboost_shadow_model.feature_count,
    )

except Exception as exc:
    _catboost_shadow_load_error = (
        f"{type(exc).__name__}: {exc}"
    )

    logger.warning(
        "CatBoostRanker shadow modeli yüklenemedi. "
        "Backend CatBoost olmadan çalışmaya devam edecek. "
        "Hata: %s",
        _catboost_shadow_load_error,
    )


def configure_matching_api(
    *,
    company_index: Any,
    reranker_model: Any | None = None,
) -> None:
    """
    Uygulama başlangıcında oluşturulan şirket indeksini
    ve Qwen reranker modelini matching endpointine bağlar.
    """

    global _company_index
    global _reranker_model

    _company_index = company_index
    _reranker_model = reranker_model


def set_company_index(
    company_index: Any,
) -> None:
    """
    Şirket indeksini sonradan bağlamak için kullanılan
    geriye dönük uyumlu yardımcı fonksiyon.
    """

    global _company_index

    _company_index = company_index


def set_reranker_model(
    reranker_model: Any,
) -> None:
    """
    Qwen reranker modelini sonradan bağlamak için kullanılan
    geriye dönük uyumlu yardımcı fonksiyon.
    """

    global _reranker_model

    _reranker_model = reranker_model


def get_company_index() -> Any | None:
    return _company_index


def get_reranker_model() -> Any | None:
    return _reranker_model


def get_catboost_shadow_model(
) -> CatBoostCompanyRanker | None:
    if not _catboost_shadow_model.is_loaded:
        return None

    return _catboost_shadow_model


def get_catboost_shadow_status() -> dict[str, Any]:
    model_status = (
        _catboost_shadow_model.get_status()
    )

    return {
        **model_status,
        "shadow_mode": True,
        "affects_ranking": False,
        "affects_decision": False,
        "startup_load_error": (
            _catboost_shadow_load_error
        ),
    }


@router.post(
    "/candidates",
    response_model=CompanyMatchResponse,
    status_code=status.HTTP_200_OK,
    summary="EFT açıklamasından şirket adaylarını bul",
)
def match_company_candidates(
    request: CompanyMatchRequest,
) -> CompanyMatchResponse:
    """
    EFT açıklamasını şirket adaylarıyla eşleştirir.

    İşlem sırası:

    1. Hibrit candidate retrieval
    2. Qwen reranker
    3. Dinamik skor birleştirme
    4. CatBoost shadow-mode değerlendirmesi
    5. AUTO_MATCH / MANUAL_REVIEW / UNKNOWN kararı
    """

    company_index = get_company_index()

    if company_index is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail=(
                "Şirket indeksi hazır değil. "
                "main.py başlangıç yapılandırmasını kontrol edin."
            ),
        )

    try:
        retrieval_result = company_index.search(
            text=request.text,
            limit=request.limit,
        )

        ranked_result = apply_company_reranking(
            result=retrieval_result,
            reranker_model=get_reranker_model(),
            rerank_limit=request.limit,
            catboost_ranker_model=(
                get_catboost_shadow_model()
            ),
        )

        return CompanyMatchResponse.model_validate(
            ranked_result
        )

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail=str(exc),
        ) from exc

    except Exception as exc:
        logger.exception(
            "Şirket eşleştirme endpointi başarısız."
        )

        raise HTTPException(
            status_code=(
                status.HTTP_500_INTERNAL_SERVER_ERROR
            ),
            detail=(
                "Şirket eşleştirme sırasında hata oluştu: "
                f"{type(exc).__name__}: {exc}"
            ),
        ) from exc