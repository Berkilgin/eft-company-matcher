from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "Qwen/Qwen3-Reranker-0.6B"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "qwen3-reranker-0.6b"
)


def model_is_complete(model_directory: Path) -> bool:
    required_files = {
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }

    existing_files = {
        path.name
        for path in model_directory.glob("*")
        if path.is_file()
    }

    has_model_weights = any(
        model_directory.glob("*.safetensors")
    )

    return (
        required_files.issubset(existing_files)
        and has_model_weights
    )


def main() -> None:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    TARGET_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if model_is_complete(TARGET_DIRECTORY):
        print("Reranker modeli zaten indirilmiş:")
        print(TARGET_DIRECTORY)
        return

    print("Qwen3 Reranker modeli indiriliyor.")
    print(f"Kaynak: {MODEL_REPOSITORY}")
    print(f"Hedef: {TARGET_DIRECTORY}")
    print("Terminali kapatmayın.")

    downloaded_path = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        local_dir=str(TARGET_DIRECTORY),
    )

    if not model_is_complete(TARGET_DIRECTORY):
        raise RuntimeError(
            "Model indirildi ancak gerekli dosyalar "
            "eksik görünüyor."
        )

    print()
    print("Reranker modeli başarıyla indirildi:")
    print(downloaded_path)


if __name__ == "__main__":
    main()