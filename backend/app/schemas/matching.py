from __future__ import annotations

import math
from typing import Any, ClassVar, Literal

from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from app.schemas.normalization import (
    NormalizationResponse,
)


MatchDecision = Literal[
    "AUTO_MATCH",
    "MANUAL_REVIEW",
    "UNKNOWN",
]


class CompanyMatchRequest(BaseModel):
    text: str = Field(
        min_length=1,
        max_length=2000,
        description=(
            "Şirket adı aranacak EFT açıklaması."
        ),
        examples=[
            "TUPRASS IZM FATURA ODEMESI",
            "ENERJISA URETIM SANTRAL ODM",
        ],
    )

    limit: int = Field(
        default=10,
        ge=1,
        le=50,
        description=(
            "Döndürülecek maksimum şirket adayı sayısı."
        ),
    )

    @field_validator("text")
    @classmethod
    def validate_text(
        cls,
        value: str,
    ) -> str:
        cleaned_value = value.strip()

        if not cleaned_value:
            raise ValueError(
                "EFT açıklaması boş bırakılamaz."
            )

        return cleaned_value


class CompanyCandidate(BaseModel):
    """
    Bir şirket adayına ait retrieval, reranker ve
    CatBoost shadow-mode özellikleri.

    0-1 aralığındaki skorlar kayan nokta hassasiyetinden
    kaynaklanan çok küçük taşmalar için doğrulama öncesinde
    güvenli biçimde kırpılır.
    """

    UNIT_INTERVAL_FIELDS: ClassVar[
        frozenset[str]
    ] = frozenset(
        {
            "identifier_score",
            "char_wb_tfidf_score",
            "raw_char_tfidf_score",
            "char_tfidf_score",
            "jaro_winkler_score",
            "token_prefix_score",
            "fuzzy_ratio_score",
            "fuzzy_partial_ratio_score",
            "fuzzy_token_set_score",
            "fuzzy_wratio_score",
            "embedding_score",
            "hybrid_score",
            "retrieval_score",
            "reranker_score",
            "adjusted_reranker_score",
            "lexical_evidence",
            "retrieval_weight",
            "reranker_weight",
            "reranker_gap",
            "candidate_reranker_gap",
            "base_reranker_weight",
            "final_score",
            "catboost_normalized_score",
        }
    )

    SCORE_EPSILON: ClassVar[float] = 1e-5

    # -----------------------------------------------------
    # Şirket ve alias bilgileri
    # -----------------------------------------------------

    company_id: int

    legal_name: str
    brand_name: str
    city: str
    sector: str

    matched_alias: str
    normalized_alias: str
    alias_type: str

    # -----------------------------------------------------
    # Deterministik eşleşme özellikleri
    # -----------------------------------------------------

    exact_match: bool
    unique_exact_match: bool
    ambiguous_exact_match: bool

    identifier_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    identifier_overlap: bool

    # -----------------------------------------------------
    # TF-IDF ve lexical özellikler
    # -----------------------------------------------------

    char_wb_tfidf_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    raw_char_tfidf_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    char_tfidf_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    jaro_winkler_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    token_prefix_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    fuzzy_ratio_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    fuzzy_partial_ratio_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    fuzzy_token_set_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    fuzzy_wratio_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    # -----------------------------------------------------
    # Embedding ve retrieval özellikleri
    # -----------------------------------------------------

    embedding_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    hybrid_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    retrieval_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    retrieval_rank: int = Field(
        default=0,
        ge=0,
    )

    # -----------------------------------------------------
    # Qwen reranker özellikleri
    # -----------------------------------------------------

    reranker_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    adjusted_reranker_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    lexical_evidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    retrieval_weight: float = Field(
        ge=0.0,
        le=1.0,
    )

    reranker_weight: float = Field(
        ge=0.0,
        le=1.0,
    )

    reranker_gap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    candidate_reranker_gap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    base_reranker_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    query_length: int = Field(
        default=0,
        ge=0,
    )

    query_token_count: int = Field(
        default=0,
        ge=0,
    )

    final_score: float = Field(
        ge=0.0,
        le=1.0,
    )

    current_final_rank: int = Field(
        default=0,
        ge=0,
    )

    reranked: bool

    # -----------------------------------------------------
    # CatBoost shadow-mode özellikleri
    # -----------------------------------------------------

    catboost_raw_score: float = 0.0

    catboost_normalized_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    catboost_rank: int = Field(
        default=0,
        ge=0,
    )

    catboost_shadow_top1: bool = False

    catboost_shadow_order_delta: int = 0

    @field_validator(
        *UNIT_INTERVAL_FIELDS,
        mode="before",
    )
    @classmethod
    def clamp_unit_interval_score(
        cls,
        value: Any,
    ) -> float:
        """
        0-1 skorlarını güvenli biçimde doğrular.

        Örnek:
            1.000001 -> 1.0
            -0.000001 -> 0.0

        Belirlenen toleranstan daha büyük taşmalar gerçek
        hesaplama hatası kabul edilir ve reddedilir.
        """

        if value is None or value == "":
            return 0.0

        if isinstance(
            value,
            bool,
        ):
            numeric_value = float(
                int(value)
            )

        else:
            try:
                numeric_value = float(
                    value
                )

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    "Skor sayısal bir değer olmalıdır."
                ) from exc

        if not math.isfinite(
            numeric_value
        ):
            raise ValueError(
                "Skor NaN veya sonsuz olamaz."
            )

        if (
            -cls.SCORE_EPSILON
            <= numeric_value
            < 0.0
        ):
            return 0.0

        if (
            1.0
            < numeric_value
            <= 1.0 + cls.SCORE_EPSILON
        ):
            return 1.0

        return numeric_value


class CompanyMatchResponse(BaseModel):
    normalization: NormalizationResponse

    query_text: str

    decision: MatchDecision

    best_candidate: CompanyCandidate | None

    score_margin: float

    candidates: list[CompanyCandidate]

    index_stats: dict[str, Any]

    # -----------------------------------------------------
    # Qwen reranker çalışma durumu
    # -----------------------------------------------------

    ranking_mode: str

    reranker_applied: bool

    reranked_candidate_count: int = Field(
        default=0,
        ge=0,
    )

    ranking_error: str | None = None

    query_length: int = Field(
        default=0,
        ge=0,
    )

    query_token_count: int = Field(
        default=0,
        ge=0,
    )

    reranker_gap: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    base_reranker_weight: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    # -----------------------------------------------------
    # CatBoost shadow-mode çalışma durumu
    # -----------------------------------------------------

    catboost_shadow_applied: bool = False

    catboost_shadow_candidate_count: int = Field(
        default=0,
        ge=0,
    )

    catboost_shadow_status: str = (
        "not_requested"
    )

    catboost_shadow_error: str | None = None

    catboost_shadow_changed_winner: bool = False

    catboost_shadow_best_company_id: int | None = None

    catboost_shadow_best_raw_score: float = 0.0

    catboost_shadow_best_normalized_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    catboost_shadow_raw_margin: float = Field(
        default=0.0,
        ge=0.0,
    )

    catboost_shadow_normalized_margin: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
    )

    @field_validator(
        "reranker_gap",
        "base_reranker_weight",
        "catboost_shadow_best_normalized_score",
        "catboost_shadow_normalized_margin",
        mode="before",
    )
    @classmethod
    def clamp_response_unit_interval_score(
        cls,
        value: Any,
    ) -> float:
        if value is None or value == "":
            return 0.0

        try:
            numeric_value = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(
                "Skor sayısal bir değer olmalıdır."
            ) from exc

        if not math.isfinite(
            numeric_value
        ):
            raise ValueError(
                "Skor NaN veya sonsuz olamaz."
            )

        epsilon = 1e-5

        if (
            -epsilon
            <= numeric_value
            < 0.0
        ):
            return 0.0

        if (
            1.0
            < numeric_value
            <= 1.0 + epsilon
        ):
            return 1.0

        return numeric_value