from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_REPOSITORY = "Qwen/Qwen3-Embedding-0.6B"

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TARGET_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "qwen3-embedding-0.6b"
)


def model_is_complete(model_directory: Path) -> bool:
    required_files = {
        "config.json",
        "tokenizer.json",
        "modules.json",
        "model.safetensors",
    }

    existing_files = {
        path.name
        for path in model_directory.glob("*")
        if path.is_file()
    }

    return required_files.issubset(existing_files)


def main() -> None:
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    TARGET_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    if model_is_complete(TARGET_DIRECTORY):
        print("Model zaten indirilmiş:")
        print(TARGET_DIRECTORY)
        return

    print("Qwen3 Embedding modeli indiriliyor.")
    print(f"Kaynak: {MODEL_REPOSITORY}")
    print(f"Hedef: {TARGET_DIRECTORY}")
    print("İndirme tamamlanana kadar terminali kapatmayın.")

    downloaded_path = snapshot_download(
        repo_id=MODEL_REPOSITORY,
        local_dir=str(TARGET_DIRECTORY),
    )

    if not model_is_complete(TARGET_DIRECTORY):
        raise RuntimeError(
            "Model indirme işlemi tamamlandı ancak "
            "gerekli model dosyaları bulunamadı."
        )

    print()
    print("Model başarıyla indirildi:")
    print(downloaded_path)


if __name__ == "__main__":
    main()