from __future__ import annotations

from pathlib import Path

import numpy as np

from app.models.embedding_model import EmbeddingModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "qwen3-embedding-0.6b"
)


def main() -> None:
    embedding_model = EmbeddingModel(
        model_path=MODEL_PATH,
        batch_size=4,
        max_sequence_length=256,
    )

    print("Model yükleniyor...")
    embedding_model.load()

    print("Model durumu:")
    print(embedding_model.get_status())

    query = "TR PET RAF IZM FATURA ODEMESI"

    documents = [
        (
            "Official company name: Türkiye Petrol "
            "Rafinerileri A.Ş. | Brand: TÜPRAŞ | "
            "Known alias: TR PETROL RAF | City: Kocaeli | "
            "Sector: Petrol ve rafineri"
        ),
        (
            "Official company name: Türkiye Petrolleri "
            "Anonim Ortaklığı | Brand: TPAO | "
            "Known alias: Türkiye Petrolleri | City: Ankara | "
            "Sector: Enerji"
        ),
        (
            "Official company name: Petrol Ofisi A.Ş. | "
            "Brand: Petrol Ofisi | Known alias: POAS | "
            "City: İstanbul | Sector: Akaryakıt"
        ),
    ]

    query_embedding = (
        embedding_model.encode_query(query)
    )

    document_embeddings = (
        embedding_model.encode_documents(documents)
    )

    similarities = (
        document_embeddings
        @ query_embedding
    )

    ranked_indices = np.argsort(
        -similarities
    )

    print()
    print(f"Sorgu: {query}")
    print()

    for rank, document_index in enumerate(
        ranked_indices,
        start=1,
    ):
        print(
            f"{rank}. skor="
            f"{similarities[document_index]:.4f}"
        )
        print(documents[document_index])
        print()

    embedding_model.unload()


if __name__ == "__main__":
    main()