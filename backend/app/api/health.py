from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_settings
from app.core.gpu import get_gpu_status
from app.schemas.health import (
    GPUStatus,
    HealthResponse,
)


router = APIRouter(
    prefix="/health",
    tags=["System"],
)

settings = get_settings()


@router.get(
    "",
    response_model=HealthResponse,
    summary="Backend ve GPU durumunu getir",
)
def health_check(
    request: Request,
) -> HealthResponse:
    gpu_status = GPUStatus.model_validate(
        get_gpu_status()
    )

    company_index = getattr(
        request.app.state,
        "company_index",
        None,
    )

    embedding_model = getattr(
        request.app.state,
        "embedding_model",
        None,
    )

    embedding_error = getattr(
        request.app.state,
        "embedding_model_error",
        None,
    )

    reranker_model = getattr(
        request.app.state,
        "reranker_model",
        None,
    )

    reranker_error = getattr(
        request.app.state,
        "reranker_model_error",
        None,
    )

    database_status = getattr(
        request.app.state,
        "database_status",
        "unknown",
    )

    database_error = getattr(
        request.app.state,
        "database_error",
        None,
    )

    database_details = getattr(
        request.app.state,
        "database_details",
        {},
    )

    data_source = getattr(
        request.app.state,
        "data_source",
        "unknown",
    )

    if company_index is None:
        company_index_status = "not_loaded"
        index_stats = {}
    else:
        company_index_status = "ready"
        index_stats = (
            company_index.get_stats()
        )

    if embedding_model is None:
        embedding_status = "not_loaded"
        embedding_details = {
            "loaded": False,
            "error": embedding_error,
        }
    else:
        embedding_status = "ready"
        embedding_details = (
            embedding_model.get_status()
        )

    if reranker_model is None:
        reranker_status = "not_loaded"
        reranker_details = {
            "loaded": False,
            "error": reranker_error,
        }
    else:
        reranker_status = "ready"
        reranker_details = (
            reranker_model.get_status()
        )

    return HealthResponse(
        status="healthy",
        application=settings.app_name,
        version=settings.app_version,
        gpu=gpu_status,
        components={
            "backend": "ready",
            "data_source": data_source,
            "database": database_status,
            "database_details": {
                **database_details,
                "error": database_error,
            },
            "company_index": (
                company_index_status
            ),
            "company_index_stats": index_stats,
            "embedding_model": (
                embedding_status
            ),
            "embedding_model_details": (
                embedding_details
            ),
            "reranker_model": (
                reranker_status
            ),
            "reranker_model_details": (
                reranker_details
            ),
            "catboost_model": "not_loaded",
        },
    )