from __future__ import annotations

from pathlib import Path

from app.models.reranker_model import (
    RerankerModel,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-reranker-0.6b"
)


def main() -> None:
    model = RerankerModel(
        model_path=MODEL_PATH,
        batch_size=2,
        max_length=256,
    )

    print("Reranker yükleniyor...")
    model.load()

    print("Model durumu:")
    print(model.get_status())

    query = (
        "Raw Turkish EFT description: "
        "TR PET RAF IZM FATURA ODEMESI | "
        "Normalized company text: "
        "tr petrol rafineri izm"
    )

    documents = [
        (
            "Official company name: Türkiye Petrol "
            "Rafinerileri A.Ş. | Brand: TÜPRAŞ | "
            "Matched alias: TR PETROL RAF | "
            "City: Kocaeli | Sector: Petrol rafinerisi"
        ),
        (
            "Official company name: Türkiye Petrolleri "
            "Anonim Ortaklığı | Brand: TPAO | "
            "Matched alias: Türkiye Petrolleri | "
            "City: Ankara | Sector: Enerji"
        ),
        (
            "Official company name: Petrol Ofisi A.Ş. | "
            "Brand: Petrol Ofisi | Matched alias: POAS | "
            "City: İstanbul | Sector: Akaryakıt"
        ),
    ]

    scores = model.score(
        query=query,
        documents=documents,
    )

    ranked_results = sorted(
        zip(scores, documents, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )

    print()
    print(f"Sorgu:\n{query}")
    print()

    for rank, (score, document) in enumerate(
        ranked_results,
        start=1,
    ):
        print(f"{rank}. Reranker skoru: {score:.6f}")
        print(document)
        print()

    model.unload()


if __name__ == "__main__":
    main()