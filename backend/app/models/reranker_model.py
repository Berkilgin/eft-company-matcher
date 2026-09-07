from __future__ import annotations

import gc
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


DEFAULT_RERANK_INSTRUCTION = (
    "Perform strict entity-name matching between the query and "
    "the candidate company record. "
    "Determine whether they refer to the same legal entity. "
    "Ignore the fact that the query may originate from a bank "
    "transfer or payment description. "
    "Do not prefer banks, financial institutions, Turkish companies, "
    "large companies, well-known companies, or popular brands. "
    "Base the decision only on company-name evidence such as character "
    "similarity, token similarity, abbreviations, known aliases, "
    "transliterations, identifiers, and plausible spelling errors. "
    "A high-level semantic relationship, the same industry, the same "
    "country, or general business relevance is not sufficient. "
    "Answer yes only when there is strong evidence that the query and "
    "candidate identify the same company."
)


SYSTEM_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document identifies the same company as the "
    "Query according to the Instruct. "
    "Use strict entity-name matching. "
    "Do not infer a match merely from industry, country, popularity, "
    "or banking context. "
    'The answer can only be "yes" or "no".'
    "<|im_end|>\n"
    "<|im_start|>user\n"
)


ASSISTANT_SUFFIX = (
    "<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


class RerankerModel:
    """
    Qwen3-Reranker-0.6B modelini yerel klasörden yükler.

    Model, her query-document çifti için 0 ile 1 arasında
    bir yeniden sıralama skoru üretir.

    Bu skor kalibre edilmiş eşleşme olasılığı değildir.
    Yalnızca adayları sıralamak amacıyla kullanılmalıdır.
    """

    def __init__(
        self,
        model_path: str | Path,
        batch_size: int = 2,
        max_length: int = 256,
    ) -> None:
        self.model_path = Path(
            model_path
        ).resolve()

        self.batch_size = batch_size
        self.max_length = max_length

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.dtype = (
            torch.float16
            if self.device == "cuda"
            else torch.float32
        )

        self.tokenizer: Any | None = None
        self.model: Any | None = None

        self.true_token_id: int | None = None
        self.false_token_id: int | None = None

        self.prefix_tokens: list[int] = []
        self.suffix_tokens: list[int] = []

        self.load_error: str | None = None

    @property
    def is_loaded(self) -> bool:
        return (
            self.model is not None
            and self.tokenizer is not None
        )

    def load(self) -> None:
        if self.is_loaded:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Reranker model klasörü bulunamadı: "
                f"{self.model_path}"
            )

        config_path = (
            self.model_path
            / "config.json"
        )

        if not config_path.exists():
            raise FileNotFoundError(
                "Reranker config.json bulunamadı: "
                f"{config_path}"
            )

        try:
            self.tokenizer = (
                AutoTokenizer.from_pretrained(
                    str(self.model_path),
                    padding_side="left",
                    local_files_only=True,
                )
            )

            if (
                self.tokenizer.pad_token_id
                is None
            ):
                self.tokenizer.pad_token = (
                    self.tokenizer.eos_token
                )

            self.model = (
                AutoModelForCausalLM
                .from_pretrained(
                    str(self.model_path),
                    torch_dtype=self.dtype,
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
            )

            self.model.to(
                self.device
            )

            self.model.eval()

            if hasattr(
                self.model.config,
                "use_cache",
            ):
                self.model.config.use_cache = (
                    False
                )

            if (
                self.model.config.pad_token_id
                is None
            ):
                self.model.config.pad_token_id = (
                    self.tokenizer.pad_token_id
                )

            self.false_token_id = (
                self._resolve_answer_token_id(
                    "no"
                )
            )

            self.true_token_id = (
                self._resolve_answer_token_id(
                    "yes"
                )
            )

            self.prefix_tokens = (
                self.tokenizer.encode(
                    SYSTEM_PREFIX,
                    add_special_tokens=False,
                )
            )

            self.suffix_tokens = (
                self.tokenizer.encode(
                    ASSISTANT_SUFFIX,
                    add_special_tokens=False,
                )
            )

            reserved_token_count = (
                len(self.prefix_tokens)
                + len(self.suffix_tokens)
            )

            if (
                self.max_length
                <= reserved_token_count
            ):
                raise ValueError(
                    "max_length, sistem prefix ve "
                    "suffix uzunluğundan büyük olmalıdır."
                )

            self.load_error = None

        except Exception as exc:
            self.load_error = (
                f"{type(exc).__name__}: {exc}"
            )

            self.model = None
            self.tokenizer = None

            self.true_token_id = None
            self.false_token_id = None

            self.prefix_tokens = []
            self.suffix_tokens = []

            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            raise

    def score(
        self,
        query: str,
        documents: Sequence[str],
        instruction: str = (
            DEFAULT_RERANK_INSTRUCTION
        ),
    ) -> list[float]:
        pairs = [
            (
                query,
                document,
            )
            for document in documents
        ]

        return self.score_pairs(
            pairs=pairs,
            instruction=instruction,
        )

    def score_pairs(
        self,
        pairs: Sequence[
            tuple[str, str]
        ],
        instruction: str = (
            DEFAULT_RERANK_INSTRUCTION
        ),
    ) -> list[float]:
        model, tokenizer = (
            self._require_model()
        )

        cleaned_instruction = (
            instruction.strip()
        )

        if not cleaned_instruction:
            raise ValueError(
                "Reranker instruction "
                "metni boş olamaz."
            )

        cleaned_pairs: list[
            tuple[str, str]
        ] = []

        for query, document in pairs:
            cleaned_query = query.strip()

            cleaned_document = (
                document.strip()
            )

            if not cleaned_query:
                raise ValueError(
                    "Reranker query metni "
                    "boş olamaz."
                )

            if not cleaned_document:
                raise ValueError(
                    "Reranker document metni "
                    "boş olamaz."
                )

            cleaned_pairs.append(
                (
                    cleaned_query,
                    cleaned_document,
                )
            )

        if not cleaned_pairs:
            return []

        all_scores: list[float] = []

        for batch_start in range(
            0,
            len(cleaned_pairs),
            self.batch_size,
        ):
            batch_pairs = cleaned_pairs[
                batch_start:
                batch_start
                + self.batch_size
            ]

            formatted_pairs = [
                self._format_instruction(
                    instruction=(
                        cleaned_instruction
                    ),
                    query=query,
                    document=document,
                )
                for query, document
                in batch_pairs
            ]

            model_inputs = (
                self._process_inputs(
                    formatted_pairs,
                    tokenizer,
                )
            )

            try:
                with torch.inference_mode():
                    outputs = model(
                        **model_inputs,
                        use_cache=False,
                    )

                    final_token_logits = (
                        outputs.logits[
                            :,
                            -1,
                            :,
                        ]
                    )

                    false_logits = (
                        final_token_logits[
                            :,
                            self.false_token_id,
                        ]
                    )

                    true_logits = (
                        final_token_logits[
                            :,
                            self.true_token_id,
                        ]
                    )

                    binary_logits = (
                        torch.stack(
                            [
                                false_logits,
                                true_logits,
                            ],
                            dim=1,
                        )
                    )

                    probabilities = (
                        torch.softmax(
                            binary_logits.float(),
                            dim=1,
                        )[:, 1]
                    )

                    batch_scores = (
                        probabilities
                        .detach()
                        .cpu()
                        .tolist()
                    )

                    all_scores.extend(
                        float(score)
                        for score
                        in batch_scores
                    )

            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "Reranker çalışırken GPU "
                    "belleği yetersiz kaldı. "
                    "batch_size değerini 1 yapın "
                    "veya max_length değerini "
                    "düşürün."
                ) from exc

        return all_scores

    def _resolve_answer_token_id(
        self,
        answer: str,
    ) -> int:
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer yüklenmemiş."
            )

        token_ids = self.tokenizer.encode(
            answer,
            add_special_tokens=False,
        )

        if not token_ids:
            raise RuntimeError(
                f"'{answer}' cevabı için "
                "token kimliği üretilemedi."
            )

        return int(
            token_ids[-1]
        )

    def _process_inputs(
        self,
        formatted_pairs: list[str],
        tokenizer: Any,
    ) -> dict[str, torch.Tensor]:
        available_text_length = (
            self.max_length
            - len(self.prefix_tokens)
            - len(self.suffix_tokens)
        )

        if available_text_length < 1:
            raise RuntimeError(
                "Reranker için kullanılabilir "
                "metin token uzunluğu sıfır."
            )

        tokenized = tokenizer(
            formatted_pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=(
                available_text_length
            ),
        )

        complete_input_ids: list[
            list[int]
        ] = []

        for token_ids in tokenized[
            "input_ids"
        ]:
            complete_token_ids = (
                self.prefix_tokens
                + token_ids
                + self.suffix_tokens
            )

            complete_input_ids.append(
                complete_token_ids
            )

        padded_inputs = tokenizer.pad(
            {
                "input_ids": (
                    complete_input_ids
                ),
            },
            padding=True,
            return_tensors="pt",
        )

        return {
            key: value.to(
                self.device
            )
            for key, value
            in padded_inputs.items()
        }

    @staticmethod
    def _format_instruction(
        instruction: str,
        query: str,
        document: str,
    ) -> str:
        return (
            f"<Instruct>: {instruction}\n"
            f"<Query>: {query}\n"
            f"<Document>: {document}"
        )

    def get_status(
        self,
    ) -> dict[str, Any]:
        return {
            "loaded": self.is_loaded,
            "model_path": str(
                self.model_path
            ),
            "device": self.device,
            "dtype": str(
                self.dtype
            ),
            "batch_size": (
                self.batch_size
            ),
            "max_length": (
                self.max_length
            ),
            "instruction_mode": (
                "strict_entity_name_matching"
            ),
            "load_error": (
                self.load_error
            ),
        }

    def unload(self) -> None:
        self.model = None
        self.tokenizer = None

        self.true_token_id = None
        self.false_token_id = None

        self.prefix_tokens = []
        self.suffix_tokens = []

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_model(
        self,
    ) -> tuple[Any, Any]:
        if (
            self.model is None
            or self.tokenizer is None
        ):
            raise RuntimeError(
                "Reranker modeli henüz "
                "yüklenmedi."
            )

        if (
            self.true_token_id is None
            or self.false_token_id is None
        ):
            raise RuntimeError(
                "Reranker yes/no token "
                "kimlikleri hazır değil."
            )

        return (
            self.model,
            self.tokenizer,
        )