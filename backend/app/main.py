from __future__ import annotations

import importlib
import inspect
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.health import (
    router as health_router,
)
from app.api.matching import (
    configure_matching_api,
    get_catboost_shadow_status,
    router as matching_router,
)
from app.api.normalization import (
    router as normalization_router,
)
from app.retrieval.company_index import (
    CompanyIndex,
)


# =========================================================
# Temel yollar
# =========================================================

BACKEND_ROOT = Path(
    __file__
).resolve().parents[1]

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
)

MODELS_ROOT = (
    PROJECT_ROOT
    / "models"
)

COMPANIES_PATH = (
    DATA_ROOT
    / "raw"
    / "companies.csv"
)

ALIASES_PATH = (
    DATA_ROOT
    / "raw"
    / "aliases.csv"
)

EMBEDDING_MODEL_PATH = (
    MODELS_ROOT
    / "qwen3-embedding-0.6b"
)

RERANKER_MODEL_PATH = (
    MODELS_ROOT
    / "qwen3-reranker-0.6b"
)


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    __name__
)


# =========================================================
# Runtime nesneleri
# =========================================================

_company_index: Any | None = None
_embedding_model: Any | None = None
_reranker_model: Any | None = None

_startup_error: str | None = None


# =========================================================
# Yardımcı fonksiyonlar
# =========================================================

def _get_component_from_health_module(
    *,
    getter_names: tuple[str, ...],
    attribute_names: tuple[str, ...],
) -> Any | None:
    """
    health.py içinde daha önceden oluşturulmuş bir çalışma
    zamanı bileşeni varsa onu döndürür.

    Böylece health.py mevcut projede indeks veya model
    oluşturuyorsa ikinci kez yükleme yapılması önlenir.
    """

    try:
        health_module = importlib.import_module(
            "app.api.health"
        )

    except Exception as exc:
        logger.debug(
            "Health modülü okunamadı: %s",
            exc,
        )

        return None

    for getter_name in getter_names:
        getter = getattr(
            health_module,
            getter_name,
            None,
        )

        if not callable(getter):
            continue

        try:
            value = getter()

        except TypeError:
            continue

        except Exception as exc:
            logger.debug(
                "Health getter başarısız: %s | %s",
                getter_name,
                exc,
            )

            continue

        if value is not None:
            return value

    for attribute_name in attribute_names:
        value = getattr(
            health_module,
            attribute_name,
            None,
        )

        if value is not None:
            return value

    return None


def _find_model_class(
    *,
    module_name: str,
    preferred_class_names: tuple[str, ...],
    required_methods: tuple[str, ...],
) -> type[Any]:
    """
    Model modülündeki uygun sınıfı bulur.

    Önce bilinen sınıf isimlerini kontrol eder. Hiçbiri
    bulunamazsa gerekli metotlara sahip yerel sınıfları
    tarar.
    """

    module = importlib.import_module(
        module_name
    )

    for class_name in preferred_class_names:
        candidate_class = getattr(
            module,
            class_name,
            None,
        )

        if (
            inspect.isclass(candidate_class)
            and all(
                callable(
                    getattr(
                        candidate_class,
                        method_name,
                        None,
                    )
                )
                for method_name
                in required_methods
            )
        ):
            return candidate_class

    for _, candidate_class in inspect.getmembers(
        module,
        inspect.isclass,
    ):
        if (
            candidate_class.__module__
            != module.__name__
        ):
            continue

        if all(
            callable(
                getattr(
                    candidate_class,
                    method_name,
                    None,
                )
            )
            for method_name
            in required_methods
        ):
            return candidate_class

    raise ImportError(
        f"{module_name} modülünde uygun model "
        "sınıfı bulunamadı. Aranan metotlar: "
        f"{required_methods}"
    )


def _instantiate_model(
    *,
    model_class: type[Any],
    model_path: Path,
) -> Any:
    """
    Farklı constructor isimlerine sahip yerel model
    sınıfları için güvenli oluşturma denemeleri yapar.
    """

    constructor_attempts: list[
        tuple[
            tuple[Any, ...],
            dict[str, Any],
        ]
    ] = [
        (
            (),
            {
                "model_path": model_path,
            },
        ),
        (
            (),
            {
                "model_path": str(
                    model_path
                ),
            },
        ),
        (
            (),
            {
                "model_dir": model_path,
            },
        ),
        (
            (),
            {
                "model_dir": str(
                    model_path
                ),
            },
        ),
        (
            (),
            {
                "path": model_path,
            },
        ),
        (
            (),
            {
                "path": str(
                    model_path
                ),
            },
        ),
        (
            (),
            {
                "model_name_or_path": str(
                    model_path
                ),
            },
        ),
        (
            (
                model_path,
            ),
            {},
        ),
        (
            (
                str(
                    model_path
                ),
            ),
            {},
        ),
        (
            (),
            {},
        ),
    ]

    errors: list[str] = []

    for (
        positional_arguments,
        keyword_arguments,
    ) in constructor_attempts:
        try:
            return model_class(
                *positional_arguments,
                **keyword_arguments,
            )

        except TypeError as exc:
            errors.append(
                str(exc)
            )

    raise TypeError(
        f"{model_class.__name__} sınıfı "
        "oluşturulamadı. Constructor denemeleri "
        f"başarısız: {errors[-5:]}"
    )


def _load_model(
    model: Any,
) -> Any:
    """
    Model nesnesini yükler.

    Desteklenen yükleme metotları:
    - load()
    - load_model()
    - initialize()
    """

    for method_name in (
        "load",
        "load_model",
        "initialize",
    ):
        load_method = getattr(
            model,
            method_name,
            None,
        )

        if not callable(load_method):
            continue

        load_method()
        break

    is_loaded = getattr(
        model,
        "is_loaded",
        None,
    )

    if (
        is_loaded is not None
        and not bool(is_loaded)
    ):
        load_error = getattr(
            model,
            "load_error",
            None,
        )

        raise RuntimeError(
            "Model yükleme işlemi tamamlandı ancak "
            "is_loaded=False. "
            f"Model: {type(model).__name__}, "
            f"Hata: {load_error}"
        )

    return model


def _load_embedding_model() -> Any:
    """
    Yerel Qwen3 embedding modelini oluşturur ve yükler.
    """

    if not EMBEDDING_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Embedding model klasörü bulunamadı: "
            f"{EMBEDDING_MODEL_PATH}"
        )

    embedding_class = _find_model_class(
        module_name=(
            "app.models.embedding_model"
        ),
        preferred_class_names=(
            "Qwen3EmbeddingModel",
            "EmbeddingModel",
            "LocalEmbeddingModel",
            "CompanyEmbeddingModel",
        ),
        required_methods=(
            "encode_query",
            "encode_documents",
        ),
    )

    embedding_model = _instantiate_model(
        model_class=embedding_class,
        model_path=EMBEDDING_MODEL_PATH,
    )

    return _load_model(
        embedding_model
    )


def _load_reranker_model() -> Any:
    """
    Yerel Qwen3 reranker modelini oluşturur ve yükler.
    """

    if not RERANKER_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Reranker model klasörü bulunamadı: "
            f"{RERANKER_MODEL_PATH}"
        )

    reranker_class = _find_model_class(
        module_name=(
            "app.models.reranker_model"
        ),
        preferred_class_names=(
            "Qwen3RerankerModel",
            "RerankerModel",
            "LocalRerankerModel",
            "CompanyRerankerModel",
        ),
        required_methods=(
            "score",
        ),
    )

    reranker_model = _instantiate_model(
        model_class=reranker_class,
        model_path=RERANKER_MODEL_PATH,
    )

    return _load_model(
        reranker_model
    )


def _build_company_index() -> CompanyIndex:
    """
    CSV dosyalarından şirket arama indeksini oluşturur.
    """

    if not COMPANIES_PATH.exists():
        raise FileNotFoundError(
            "Şirket CSV dosyası bulunamadı: "
            f"{COMPANIES_PATH}"
        )

    if not ALIASES_PATH.exists():
        raise FileNotFoundError(
            "Alias CSV dosyası bulunamadı: "
            f"{ALIASES_PATH}"
        )

    logger.info(
        "Şirket indeksi oluşturuluyor."
    )

    company_index = CompanyIndex.from_csv(
        companies_path=COMPANIES_PATH,
        aliases_path=ALIASES_PATH,
    )

    stats = company_index.get_stats()

    logger.info(
        "Şirket indeksi oluşturuldu. "
        "Şirket: %s | Alias: %s",
        stats.get(
            "company_count"
        ),
        stats.get(
            "alias_count"
        ),
    )

    return company_index


def _attach_embedding_model(
    *,
    company_index: Any,
    embedding_model: Any,
) -> None:
    """
    Embedding modelini şirket indeksine bağlar.
    """

    attach_method = getattr(
        company_index,
        "attach_embedding_model",
        None,
    )

    if not callable(attach_method):
        raise AttributeError(
            "Şirket indeksinde "
            "attach_embedding_model metodu yok."
        )

    already_attached_model = getattr(
        company_index,
        "embedding_model",
        None,
    )

    if already_attached_model is embedding_model:
        return

    logger.info(
        "Embedding matrisi hazırlanıyor."
    )

    attach_method(
        embedding_model
    )

    logger.info(
        "Embedding modeli şirkete indeksine bağlandı."
    )


def _publish_runtime_to_health_module(
    *,
    company_index: Any,
    embedding_model: Any,
    reranker_model: Any,
) -> None:
    """
    Runtime bileşenlerini health.py ile paylaşır.

    Health modülü farklı setter isimleri kullanıyorsa
    desteklenen yaygın isimler denenir.
    """

    try:
        health_module = importlib.import_module(
            "app.api.health"
        )

    except Exception as exc:
        logger.warning(
            "Runtime bileşenleri health modülüne "
            "aktarılamadı: %s",
            exc,
        )

        return

    component_setters = {
        "company_index": (
            company_index,
            (
                "set_company_index",
                "configure_company_index",
            ),
        ),
        "embedding_model": (
            embedding_model,
            (
                "set_embedding_model",
                "configure_embedding_model",
            ),
        ),
        "reranker_model": (
            reranker_model,
            (
                "set_reranker_model",
                "configure_reranker_model",
            ),
        ),
    }

    for (
        component_name,
        (
            component_value,
            setter_names,
        ),
    ) in component_setters.items():
        for setter_name in setter_names:
            setter = getattr(
                health_module,
                setter_name,
                None,
            )

            if not callable(setter):
                continue

            try:
                setter(
                    component_value
                )

            except Exception as exc:
                logger.debug(
                    "%s başarısız: %s",
                    setter_name,
                    exc,
                )

            break

        common_attribute_names = (
            component_name,
            f"_{component_name}",
            component_name.upper(),
        )

        for attribute_name in common_attribute_names:
            if hasattr(
                health_module,
                attribute_name,
            ):
                try:
                    setattr(
                        health_module,
                        attribute_name,
                        component_value,
                    )

                except Exception:
                    pass

    for configure_name in (
        "configure_health_api",
        "configure_runtime",
        "configure_components",
        "set_runtime_components",
    ):
        configure_function = getattr(
            health_module,
            configure_name,
            None,
        )

        if not callable(
            configure_function
        ):
            continue

        try:
            signature = inspect.signature(
                configure_function
            )

            available_values = {
                "company_index": company_index,
                "embedding_model": (
                    embedding_model
                ),
                "reranker_model": (
                    reranker_model
                ),
            }

            keyword_arguments = {
                parameter_name: (
                    available_values[
                        parameter_name
                    ]
                )
                for parameter_name
                in signature.parameters
                if parameter_name
                in available_values
            }

            configure_function(
                **keyword_arguments
            )

        except Exception as exc:
            logger.debug(
                "%s çağrısı başarısız: %s",
                configure_name,
                exc,
            )

        break


def _safe_unload(
    component: Any | None,
    component_name: str,
) -> None:
    """
    Uygulama kapanırken model kaynağını güvenli biçimde
    serbest bırakır.
    """

    if component is None:
        return

    for method_name in (
        "unload",
        "close",
        "shutdown",
    ):
        unload_method = getattr(
            component,
            method_name,
            None,
        )

        if not callable(unload_method):
            continue

        try:
            unload_method()

            logger.info(
                "%s kapatıldı.",
                component_name,
            )

        except Exception as exc:
            logger.warning(
                "%s kapatılırken hata oluştu: %s",
                component_name,
                exc,
            )

        return


# =========================================================
# Uygulama yaşam döngüsü
# =========================================================

@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> Iterator[None]:
    global _company_index
    global _embedding_model
    global _reranker_model
    global _startup_error

    logger.info(
        "EFT Company Matcher başlatılıyor."
    )

    try:
        # Health modülünde önceden oluşturulan nesneler
        # varsa tekrar oluşturma.
        _company_index = (
            _get_component_from_health_module(
                getter_names=(
                    "get_company_index",
                    "get_index",
                ),
                attribute_names=(
                    "company_index",
                    "_company_index",
                    "COMPANY_INDEX",
                ),
            )
        )

        _embedding_model = (
            _get_component_from_health_module(
                getter_names=(
                    "get_embedding_model",
                    "get_company_embedding_model",
                ),
                attribute_names=(
                    "embedding_model",
                    "_embedding_model",
                    "EMBEDDING_MODEL",
                ),
            )
        )

        _reranker_model = (
            _get_component_from_health_module(
                getter_names=(
                    "get_reranker_model",
                    "get_company_reranker",
                ),
                attribute_names=(
                    "reranker_model",
                    "_reranker_model",
                    "RERANKER_MODEL",
                ),
            )
        )

        if _company_index is None:
            _company_index = (
                _build_company_index()
            )

        if _embedding_model is None:
            logger.info(
                "Qwen3 embedding modeli yükleniyor."
            )

            _embedding_model = (
                _load_embedding_model()
            )

            logger.info(
                "Embedding modeli yüklendi: %s",
                type(
                    _embedding_model
                ).__name__,
            )

        index_embedding_model = getattr(
            _company_index,
            "embedding_model",
            None,
        )

        if (
            index_embedding_model
            is None
        ):
            _attach_embedding_model(
                company_index=_company_index,
                embedding_model=(
                    _embedding_model
                ),
            )

        if _reranker_model is None:
            logger.info(
                "Qwen3 reranker modeli yükleniyor."
            )

            _reranker_model = (
                _load_reranker_model()
            )

            logger.info(
                "Reranker modeli yüklendi: %s",
                type(
                    _reranker_model
                ).__name__,
            )

        configure_matching_api(
            company_index=_company_index,
            reranker_model=_reranker_model,
        )

        _publish_runtime_to_health_module(
            company_index=_company_index,
            embedding_model=_embedding_model,
            reranker_model=_reranker_model,
        )

        app.state.company_index = (
            _company_index
        )

        app.state.embedding_model = (
            _embedding_model
        )

        app.state.reranker_model = (
            _reranker_model
        )

        app.state.startup_error = None

        _startup_error = None

        index_stats = (
            _company_index.get_stats()
            if callable(
                getattr(
                    _company_index,
                    "get_stats",
                    None,
                )
            )
            else {}
        )

        catboost_status = (
            get_catboost_shadow_status()
        )

        logger.info(
            "EFT Company Matcher hazır. "
            "Şirket: %s | Alias: %s | "
            "Embedding: %s | Reranker: %s | "
            "CatBoost shadow: %s",
            index_stats.get(
                "company_count"
            ),
            index_stats.get(
                "alias_count"
            ),
            type(
                _embedding_model
            ).__name__,
            type(
                _reranker_model
            ).__name__,
            catboost_status.get(
                "loaded"
            ),
        )

    except Exception as exc:
        _startup_error = (
            f"{type(exc).__name__}: {exc}"
        )

        app.state.startup_error = (
            _startup_error
        )

        logger.exception(
            "Backend başlangıç işlemi başarısız: %s",
            _startup_error,
        )

        # Başlangıç hatasında uygulamayı sessizce ve yarım
        # durumda çalıştırmak yerine Uvicorn başlangıcını
        # durdur.
        raise

    try:
        yield

    finally:
        logger.info(
            "EFT Company Matcher kapatılıyor."
        )

        _safe_unload(
            _reranker_model,
            "Reranker modeli",
        )

        _safe_unload(
            _embedding_model,
            "Embedding modeli",
        )

        _company_index = None
        _embedding_model = None
        _reranker_model = None


# =========================================================
# FastAPI uygulaması
# =========================================================

app = FastAPI(
    title="EFT Company Matcher API",
    description=(
        "EFT açıklamalarından tüzel şirket adayı "
        "tespit eden yerel eşleştirme API'si."
    ),
    version="0.6.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=[
        "*",
    ],
    allow_headers=[
        "*",
    ],
)


# =========================================================
# Router kayıtları
# =========================================================

API_PREFIX = "/api/v1"


app.include_router(
    health_router,
    prefix=API_PREFIX,
)

app.include_router(
    normalization_router,
    prefix=API_PREFIX,
)

app.include_router(
    matching_router,
    prefix=API_PREFIX,
)

# =========================================================
# Temel endpointler
# =========================================================

@app.get(
    "/",
    tags=[
        "Application",
    ],
)
def read_root() -> dict[str, Any]:
    return {
        "app": "EFT Company Matcher",
        "version": "0.6.0",
        "status": (
            "healthy"
            if _startup_error is None
            else "degraded"
        ),
        "docs": "/docs",
        "health": "/api/v1/health",
        "matching": (
            "/api/v1/matching/candidates"
        ),
        "startup_error": _startup_error,
    }


@app.get(
    "/api/v1/runtime",
    tags=[
        "Application",
    ],
)
def read_runtime() -> dict[str, Any]:
    index_stats: dict[str, Any] = {}

    if (
        _company_index is not None
        and callable(
            getattr(
                _company_index,
                "get_stats",
                None,
            )
        )
    ):
        index_stats = (
            _company_index.get_stats()
        )

    return {
        "ready": (
            _company_index is not None
            and _embedding_model is not None
            and _reranker_model is not None
            and _startup_error is None
        ),
        "company_index": {
            "loaded": (
                _company_index
                is not None
            ),
            "type": (
                type(
                    _company_index
                ).__name__
                if _company_index
                is not None
                else None
            ),
            "stats": index_stats,
        },
        "embedding_model": {
            "loaded": bool(
                getattr(
                    _embedding_model,
                    "is_loaded",
                    (
                        _embedding_model
                        is not None
                    ),
                )
            ),
            "type": (
                type(
                    _embedding_model
                ).__name__
                if _embedding_model
                is not None
                else None
            ),
        },
        "reranker_model": {
            "loaded": bool(
                getattr(
                    _reranker_model,
                    "is_loaded",
                    (
                        _reranker_model
                        is not None
                    ),
                )
            ),
            "type": (
                type(
                    _reranker_model
                ).__name__
                if _reranker_model
                is not None
                else None
            ),
        },
        "catboost_shadow": (
            get_catboost_shadow_status()
        ),
        "startup_error": (
            _startup_error
        ),
    }