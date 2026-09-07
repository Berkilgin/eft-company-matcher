from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from catboost import CatBoostRanker


PROJECT_ROOT = Path(
    __file__
).resolve().parents[3]


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "catboost"
    / "company_ranker.cbm"
)

DEFAULT_METADATA_PATH = (
    PROJECT_ROOT
    / "models"
    / "catboost"
    / "company_ranker_metadata.json"
)


MODEL_PATH_ENV_NAMES: tuple[str, ...] = (
    "EFT_CATBOOST_MODEL_PATH",
    "CATBOOST_RANKER_MODEL_PATH",
)

METADATA_PATH_ENV_NAMES: tuple[str, ...] = (
    "EFT_CATBOOST_METADATA_PATH",
    "CATBOOST_RANKER_METADATA_PATH",
)


def _first_environment_value(
    names: Sequence[str],
) -> tuple[str | None, str | None]:
    """
    Tanımlı ilk ortam değişkeninin adını ve değerini döndürür.
    """

    for name in names:
        value = os.getenv(
            name
        )

        if value is None:
            continue

        cleaned_value = value.strip()

        if cleaned_value:
            return (
                name,
                cleaned_value,
            )

    return (
        None,
        None,
    )


def _resolve_path(
    value: str | Path,
) -> Path:
    """
    Mutlak yolları doğrudan, göreli yolları proje köküne
    göre çözümler.
    """

    path = Path(
        value
    ).expanduser()

    if not path.is_absolute():
        path = (
            PROJECT_ROOT
            / path
        )

    return path.resolve()


def _safe_float(
    value: Any,
    default: float = 0.0,
) -> float:
    """
    Aday sözlüğündeki feature değerini CatBoost için
    güvenli sayısal değere dönüştürür.
    """

    if value is None:
        return default

    if isinstance(
        value,
        bool,
    ):
        return float(
            int(value)
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric_value = float(
            value
        )

    else:
        text = str(
            value
        ).strip()

        if not text:
            return default

        try:
            numeric_value = float(
                text.replace(
                    ",",
                    ".",
                )
            )

        except (
            TypeError,
            ValueError,
        ):
            return default

    if not math.isfinite(
        numeric_value
    ):
        return default

    return numeric_value


class CatBoostCompanyRanker:
    """
    Eğitilmiş CatBoostRanker modelini yükleyen ve şirket
    adayları için shadow-mode skorları üreten wrapper.

    Model yolu öncelik sırası:

    1. Constructor model_path parametresi
    2. EFT_CATBOOST_MODEL_PATH
    3. CATBOOST_RANKER_MODEL_PATH
    4. Varsayılan company_ranker.cbm

    Metadata yolu için de aynı öncelik uygulanır.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
    ) -> None:
        (
            model_environment_name,
            model_environment_value,
        ) = _first_environment_value(
            MODEL_PATH_ENV_NAMES
        )

        (
            metadata_environment_name,
            metadata_environment_value,
        ) = _first_environment_value(
            METADATA_PATH_ENV_NAMES
        )

        resolved_model_value: str | Path = (
            model_path
            if model_path is not None
            else (
                model_environment_value
                if model_environment_value
                is not None
                else DEFAULT_MODEL_PATH
            )
        )

        resolved_metadata_value: str | Path = (
            metadata_path
            if metadata_path is not None
            else (
                metadata_environment_value
                if metadata_environment_value
                is not None
                else DEFAULT_METADATA_PATH
            )
        )

        self.model_path = _resolve_path(
            resolved_model_value
        )

        self.metadata_path = _resolve_path(
            resolved_metadata_value
        )

        self.model_path_source = (
            "constructor"
            if model_path is not None
            else (
                model_environment_name
                if model_environment_value
                is not None
                else "default"
            )
        )

        self.metadata_path_source = (
            "constructor"
            if metadata_path is not None
            else (
                metadata_environment_name
                if metadata_environment_value
                is not None
                else "default"
            )
        )

        self.model: CatBoostRanker | None = None

        self.metadata: dict[str, Any] = {}

        self.feature_columns: list[str] = []

        self.load_error: str | None = None

    @property
    def is_loaded(
        self,
    ) -> bool:
        return (
            self.model is not None
            and bool(
                self.feature_columns
            )
        )

    @property
    def loaded(
        self,
    ) -> bool:
        """
        Eski kodlarla geriye dönük uyumlu alias.
        """

        return self.is_loaded

    @property
    def feature_count(
        self,
    ) -> int:
        return len(
            self.feature_columns
        )

    def load(
        self,
    ) -> None:
        """
        Metadata ve CatBoost model dosyasını yükler.

        Model ile metadata feature sırası birebir aynı
        olmalıdır.
        """

        self.model = None
        self.metadata = {}
        self.feature_columns = []
        self.load_error = None

        try:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    "CatBoost model dosyası bulunamadı: "
                    f"{self.model_path}"
                )

            if not self.metadata_path.exists():
                raise FileNotFoundError(
                    "CatBoost metadata dosyası bulunamadı: "
                    f"{self.metadata_path}"
                )

            metadata_payload = json.loads(
                self.metadata_path.read_text(
                    encoding="utf-8-sig",
                )
            )

            if not isinstance(
                metadata_payload,
                dict,
            ):
                raise TypeError(
                    "CatBoost metadata kök değeri sözlük olmalıdır."
                )

            feature_columns = metadata_payload.get(
                "feature_columns"
            )

            if not isinstance(
                feature_columns,
                list,
            ):
                raise ValueError(
                    "Metadata içinde feature_columns listesi bulunamadı."
                )

            cleaned_feature_columns = [
                str(
                    feature_name
                ).strip()
                for feature_name
                in feature_columns
                if str(
                    feature_name
                ).strip()
            ]

            if not cleaned_feature_columns:
                raise ValueError(
                    "Metadata feature_columns listesi boş."
                )

            duplicate_features = sorted(
                {
                    feature_name
                    for feature_name
                    in cleaned_feature_columns
                    if cleaned_feature_columns.count(
                        feature_name
                    ) > 1
                }
            )

            if duplicate_features:
                raise ValueError(
                    "Metadata içinde tekrarlanan feature "
                    f"isimleri var: {duplicate_features}"
                )

            model = CatBoostRanker()

            model.load_model(
                str(
                    self.model_path
                )
            )

            if not model.is_fitted():
                raise RuntimeError(
                    "CatBoost model dosyası yüklendi ancak "
                    "model fitted durumda değil."
                )

            model_feature_names = [
                str(
                    feature_name
                ).strip()
                for feature_name
                in (
                    model.feature_names_
                    or []
                )
            ]

            if (
                model_feature_names
                and model_feature_names
                != cleaned_feature_columns
            ):
                raise ValueError(
                    "Model feature sırası ile metadata feature "
                    "sırası eşleşmiyor. "
                    f"Model: {model_feature_names}, "
                    f"Metadata: {cleaned_feature_columns}"
                )

            metadata_feature_count = metadata_payload.get(
                "feature_count"
            )

            if (
                metadata_feature_count is not None
                and int(
                    metadata_feature_count
                )
                != len(
                    cleaned_feature_columns
                )
            ):
                raise ValueError(
                    "Metadata feature_count değeri ile "
                    "feature_columns uzunluğu eşleşmiyor. "
                    f"feature_count={metadata_feature_count}, "
                    "feature_columns="
                    f"{len(cleaned_feature_columns)}"
                )

            self.model = model
            self.metadata = metadata_payload
            self.feature_columns = (
                cleaned_feature_columns
            )

        except Exception as exc:
            self.model = None
            self.metadata = {}
            self.feature_columns = []

            self.load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            raise

    def unload(
        self,
    ) -> None:
        self.model = None
        self.metadata = {}
        self.feature_columns = []
        self.load_error = None

    def _require_loaded_model(
        self,
    ) -> CatBoostRanker:
        if not self.is_loaded:
            raise RuntimeError(
                "CatBoostRanker modeli yüklü değil. "
                f"Son yükleme hatası: {self.load_error}"
            )

        assert self.model is not None

        return self.model

    def build_feature_matrix(
        self,
        candidates: Sequence[
            Mapping[str, Any]
        ],
    ) -> np.ndarray:
        """
        Aday sözlüklerini metadata içindeki feature sırasına
        göre sayısal matrise dönüştürür.
        """

        if not self.feature_columns:
            raise RuntimeError(
                "Feature kolonları yüklenmemiş."
            )

        if not candidates:
            return np.empty(
                (
                    0,
                    self.feature_count,
                ),
                dtype=np.float64,
            )

        rows: list[list[float]] = []

        for candidate_index, candidate in enumerate(
            candidates,
            start=1,
        ):
            if not isinstance(
                candidate,
                Mapping,
            ):
                raise TypeError(
                    "CatBoost adayı sözlük yapısında değil. "
                    f"Aday sıra: {candidate_index}, "
                    f"Tür: {type(candidate).__name__}"
                )

            row = [
                _safe_float(
                    candidate.get(
                        feature_name,
                        0.0,
                    )
                )
                for feature_name
                in self.feature_columns
            ]

            rows.append(
                row
            )

        matrix = np.asarray(
            rows,
            dtype=np.float64,
        )

        expected_shape = (
            len(candidates),
            self.feature_count,
        )

        if matrix.shape != expected_shape:
            raise ValueError(
                "CatBoost feature matrisi beklenen boyutta değil. "
                f"Beklenen: {expected_shape}, "
                f"Gerçek: {matrix.shape}"
            )

        if not np.isfinite(
            matrix
        ).all():
            raise ValueError(
                "CatBoost feature matrisinde NaN veya "
                "sonsuz değer bulundu."
            )

        return matrix

    def predict_raw_scores(
        self,
        candidates: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[float]:
        """
        Her aday için ham CatBoost ranking skoru üretir.
        """

        model = self._require_loaded_model()

        if not candidates:
            return []

        feature_matrix = self.build_feature_matrix(
            candidates
        )

        predictions = np.asarray(
            model.predict(
                feature_matrix
            ),
            dtype=np.float64,
        ).reshape(-1)

        if len(predictions) != len(
            candidates
        ):
            raise RuntimeError(
                "CatBoost tahmin sayısı aday sayısıyla "
                "eşleşmiyor. "
                f"Tahmin: {len(predictions)}, "
                f"Aday: {len(candidates)}"
            )

        if not np.isfinite(
            predictions
        ).all():
            raise RuntimeError(
                "CatBoost tahminlerinde NaN veya "
                "sonsuz değer bulundu."
            )

        return [
            float(
                score
            )
            for score in predictions
        ]

    @staticmethod
    def normalize_scores(
        raw_scores: Sequence[float],
    ) -> list[float]:
        """
        Aynı sorguya ait CatBoost skorlarını min-max
        yöntemiyle 0-1 aralığına dönüştürür.

        Bütün skorlar eşitse adaylar arasında ayrım olmadığı
        için her adaya 0.5 atanır.
        """

        if not raw_scores:
            return []

        values = np.asarray(
            raw_scores,
            dtype=np.float64,
        ).reshape(-1)

        if not np.isfinite(
            values
        ).all():
            raise ValueError(
                "Normalize edilecek CatBoost skorlarında "
                "NaN veya sonsuz değer var."
            )

        if len(values) == 1:
            return [
                1.0
            ]

        minimum_value = float(
            values.min()
        )

        maximum_value = float(
            values.max()
        )

        score_range = (
            maximum_value
            - minimum_value
        )

        if score_range <= 1e-12:
            return [
                0.5
                for _ in values
            ]

        normalized_values = (
            values
            - minimum_value
        ) / score_range

        normalized_values = np.clip(
            normalized_values,
            0.0,
            1.0,
        )

        return [
            float(
                score
            )
            for score in normalized_values
        ]

    def predict_normalized_scores(
        self,
        candidates: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[float]:
        raw_scores = self.predict_raw_scores(
            candidates
        )

        return self.normalize_scores(
            raw_scores
        )

    def predict_scores(
        self,
        candidates: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[float]:
        """
        Geriye dönük uyumlu ham skor alias'ı.
        """

        return self.predict_raw_scores(
            candidates
        )

    def score_candidates(
        self,
        candidates: Sequence[
            Mapping[str, Any]
        ],
    ) -> list[dict[str, float]]:
        """
        Ham ve normalize skorları birlikte döndürür.
        """

        raw_scores = self.predict_raw_scores(
            candidates
        )

        normalized_scores = (
            self.normalize_scores(
                raw_scores
            )
        )

        return [
            {
                "raw_score": raw_score,
                "normalized_score": (
                    normalized_score
                ),
            }
            for (
                raw_score,
                normalized_score,
            ) in zip(
                raw_scores,
                normalized_scores,
                strict=True,
            )
        ]

    def get_status(
        self,
    ) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_type": (
                type(
                    self.model
                ).__name__
                if self.model is not None
                else "CatBoostRanker"
            ),
            "model_path": str(
                self.model_path
            ),
            "metadata_path": str(
                self.metadata_path
            ),
            "model_path_source": (
                self.model_path_source
            ),
            "metadata_path_source": (
                self.metadata_path_source
            ),
            "feature_count": (
                self.feature_count
            ),
            "feature_columns": list(
                self.feature_columns
            ),
            "model_version": (
                self.metadata.get(
                    "model_version"
                )
            ),
            "created_at": self.metadata.get(
                "created_at"
            ),
            "actual_task_type": (
                self.metadata.get(
                    "training",
                    {},
                ).get(
                    "actual_task_type"
                )
                if self.metadata
                else None
            ),
            "best_iteration": (
                self.metadata.get(
                    "training",
                    {},
                ).get(
                    "best_iteration"
                )
                if self.metadata
                else None
            ),
            "load_error": self.load_error,
        }