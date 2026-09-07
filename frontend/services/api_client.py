from __future__ import annotations

from typing import Any

import requests


DEFAULT_BACKEND_URL = "http://127.0.0.1:8000"


class BackendConnectionError(RuntimeError):
    """Backend servisine erişilemediğinde oluşturulan hata."""


def _extract_error_detail(response: requests.Response) -> str:
    """Backend hata cevabından okunabilir mesaj çıkarır."""
    try:
        error_json = response.json()
        detail = error_json.get("detail", response.text)

        if isinstance(detail, list):
            return "; ".join(
                str(item.get("msg", item))
                if isinstance(item, dict)
                else str(item)
                for item in detail
            )

        return str(detail)

    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def get_backend_health(
    backend_url: str = DEFAULT_BACKEND_URL,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    """Backend, GPU ve model durumunu getirir."""

    endpoint = f"{backend_url.rstrip('/')}/api/v1/health"

    try:
        response = requests.get(
            endpoint,
            timeout=timeout_seconds,
        )
        response.raise_for_status()

    except requests.Timeout as exc:
        raise BackendConnectionError(
            "Backend durum isteği zaman aşımına uğradı."
        ) from exc

    except requests.ConnectionError as exc:
        raise BackendConnectionError(
            "FastAPI backend servisine bağlanılamadı. "
            "Backend'in 8000 portunda çalıştığını kontrol edin."
        ) from exc

    except requests.HTTPError as exc:
        raise BackendConnectionError(
            "Backend durum sorgusu başarısız: "
            f"{_extract_error_detail(response)}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise BackendConnectionError(
            "Backend geçerli bir JSON cevabı döndürmedi."
        ) from exc


def normalize_eft_description(
    text: str,
    backend_url: str = DEFAULT_BACKEND_URL,
    timeout_seconds: float = 10.0,
) -> dict[str, Any]:
    """EFT açıklamasını backend üzerinden normalize eder."""

    endpoint = (
        f"{backend_url.rstrip('/')}"
        "/api/v1/normalization/normalize"
    )

    try:
        response = requests.post(
            endpoint,
            json={"text": text},
            timeout=timeout_seconds,
        )
        response.raise_for_status()

    except requests.Timeout as exc:
        raise BackendConnectionError(
            "Normalizasyon isteği zaman aşımına uğradı."
        ) from exc

    except requests.ConnectionError as exc:
        raise BackendConnectionError(
            "FastAPI backend servisine bağlanılamadı."
        ) from exc

    except requests.HTTPError as exc:
        raise BackendConnectionError(
            "Normalizasyon işlemi başarısız: "
            f"{_extract_error_detail(response)}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise BackendConnectionError(
            "Backend geçerli bir JSON cevabı döndürmedi."
        ) from exc


def match_eft_description(
    text: str,
    limit: int = 10,
    backend_url: str = DEFAULT_BACKEND_URL,
    timeout_seconds: float = 15.0,
) -> dict[str, Any]:
    """EFT açıklaması için en olası şirket adaylarını getirir."""

    endpoint = (
        f"{backend_url.rstrip('/')}"
        "/api/v1/matching/candidates"
    )

    try:
        response = requests.post(
            endpoint,
            json={
                "text": text,
                "limit": limit,
            },
            timeout=timeout_seconds,
        )
        response.raise_for_status()

    except requests.Timeout as exc:
        raise BackendConnectionError(
            "Şirket eşleştirme isteği zaman aşımına uğradı."
        ) from exc

    except requests.ConnectionError as exc:
        raise BackendConnectionError(
            "FastAPI backend servisine bağlanılamadı. "
            "Backend'in 8000 portunda çalıştığını kontrol edin."
        ) from exc

    except requests.HTTPError as exc:
        raise BackendConnectionError(
            "Şirket eşleştirmesi başarısız: "
            f"{_extract_error_detail(response)}"
        ) from exc

    try:
        return response.json()

    except ValueError as exc:
        raise BackendConnectionError(
            "Backend geçerli bir JSON cevabı döndürmedi."
        ) from exc