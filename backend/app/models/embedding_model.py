from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


QUERY_INSTRUCTION = (
    "Retrieve the company record that corresponds to the incomplete, "
    "misspelled, abbreviated, or altered company name in a Turkish "
    "bank transfer description."
)

QUERY_PROMPT = (
    f"Instruct: {QUERY_INSTRUCTION}\n"
    "Query: "
)


class EmbeddingModel:
    """
    Qwen3 Embedding modelini yerel klasörden yükler ve çalıştırır.

    Model:
        Qwen/Qwen3-Embedding-0.6B

    GPU varsa FP16 CUDA kullanılır.
    GPU yoksa CPU ve FP32 kullanılır.
    """

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 8,
        max_sequence_length: int = 256,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.torch_dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.model: SentenceTransformer | None = None
        self.embedding_dimension: int = 0
        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if self.is_loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Embedding model klasörü bulunamadı: "
                f"{self.model_path}"
            )

        config_path = self.model_path / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(
                "Embedding modelinin config.json dosyası bulunamadı: "
                f"{config_path}"
            )

        try:
            model_kwargs: dict[str, Any] = {
                "torch_dtype": self.torch_dtype,
            }

            self.model = SentenceTransformer(
                str(self.model_path),
                device=self.device,
                model_kwargs=model_kwargs,
                tokenizer_kwargs={
                    "padding_side": "left",
                },
            )

            self.model.max_seq_length = (
                self.max_sequence_length
            )

            self.model.eval()

            dimension = (
                self.model.get_sentence_embedding_dimension()
            )

            if dimension is None:
                raise RuntimeError(
                    "Model embedding boyutunu döndürmedi."
                )

            self.embedding_dimension = int(dimension)
            self.load_error = None

        except Exception as exc:
            self.model = None
            self.embedding_dimension = 0
            self.load_error = (
                f"{type(exc).__name__}: {exc}"
            )
            raise

    def encode_query(
        self,
        text: str,
    ) -> np.ndarray:
        """
        EFT açıklaması için instruction içeren query embedding üretir.
        """
        model = self._require_model()

        cleaned_text = text.strip()

        if not cleaned_text:
            raise ValueError(
                "Embedding üretilecek sorgu boş olamaz."
            )

        with torch.inference_mode():
            embedding = model.encode(
                [cleaned_text],
                prompt=QUERY_PROMPT,
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )

        result = np.asarray(
            embedding[0],
            dtype=np.float32,
        )

        return result

    def encode_documents(
        self,
        texts: Sequence[str],
    ) -> np.ndarray:
        """
        Şirket ve alias kayıtları için document embeddingleri üretir.

        Document tarafına query instruction eklenmez.
        """
        model = self._require_model()

        cleaned_texts = [
            text.strip()
            for text in texts
            if text and text.strip()
        ]

        if not cleaned_texts:
            raise ValueError(
                "Embedding üretilecek belge listesi boş olamaz."
            )

        with torch.inference_mode():
            embeddings = model.encode(
                cleaned_texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
                device=self.device,
            )

        return np.asarray(
            embeddings,
            dtype=np.float32,
        )

    def get_status(self) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_path": str(self.model_path),
            "device": self.device,
            "dtype": str(self.torch_dtype),
            "embedding_dimension": (
                self.embedding_dimension
            ),
            "batch_size": self.batch_size,
            "max_sequence_length": (
                self.max_sequence_length
            ),
            "load_error": self.load_error,
        }

    def unload(self) -> None:
        self.model = None
        self.embedding_dimension = 0

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(self) -> SentenceTransformer:
        if self.model is None:
            raise RuntimeError(
                "Embedding modeli henüz yüklenmedi."
            )

        return self.model