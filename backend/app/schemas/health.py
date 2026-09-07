from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class GPUStatus(BaseModel):
    model_config = ConfigDict(extra="allow")

    torch_version: str
    cuda_available: bool
    cuda_version: str | None = None
    cudnn_version: int | None = None

    device: str
    device_count: int = 0

    gpu_name: str | None = None
    gpu_index: int | None = None

    total_memory_gb: float | None = None
    allocated_memory_gb: float | None = None
    reserved_memory_gb: float | None = None

    compute_capability: str | None = None
    error: str | None = None


class HealthResponse(BaseModel):
    status: str = Field(description="Backend servis durumu")
    application: str
    version: str
    gpu: GPUStatus
    components: dict[str, Any]