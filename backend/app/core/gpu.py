from __future__ import annotations

from typing import Any

import torch


GIBIBYTE = 1024**3


def bytes_to_gib(value: int) -> float:
    """Byte değerini GiB değerine dönüştürür."""
    return round(value / GIBIBYTE, 2)


def get_gpu_status() -> dict[str, Any]:
    """
    PyTorch ve CUDA durumunu güvenli biçimde döndürür.

    Bu fonksiyon CUDA bulunmadığında hata fırlatmaz. CPU bilgisi döndürerek
    uygulamanın çalışmaya devam etmesini sağlar.
    """
    status: dict[str, Any] = {
        "torch_version": torch.__version__,
        "cuda_available": False,
        "cuda_version": torch.version.cuda,
        "cudnn_version": None,
        "device": "cpu",
        "device_count": 0,
        "gpu_name": None,
        "gpu_index": None,
        "total_memory_gb": None,
        "allocated_memory_gb": None,
        "reserved_memory_gb": None,
        "compute_capability": None,
        "error": None,
    }

    try:
        cuda_available = torch.cuda.is_available()
        status["cuda_available"] = cuda_available

        if not cuda_available:
            return status

        device_index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(device_index)

        status.update(
            {
                "device": f"cuda:{device_index}",
                "device_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(device_index),
                "gpu_index": device_index,
                "total_memory_gb": bytes_to_gib(properties.total_memory),
                "allocated_memory_gb": bytes_to_gib(
                    torch.cuda.memory_allocated(device_index)
                ),
                "reserved_memory_gb": bytes_to_gib(
                    torch.cuda.memory_reserved(device_index)
                ),
                "compute_capability": (
                    f"{properties.major}.{properties.minor}"
                ),
                "cudnn_version": torch.backends.cudnn.version(),
            }
        )

    except Exception as exc:
        status["error"] = (
            f"GPU bilgileri okunurken hata oluştu: "
            f"{type(exc).__name__}: {exc}"
        )

    return status