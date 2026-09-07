from __future__ import annotations

import streamlit as st

from services.api_client import (
    BackendConnectionError,
    get_backend_health,
)


st.set_page_config(
    page_title="EFT Company Matcher",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.title("EFT Company Matcher")
st.caption(
    "EFT açıklamalarındaki eksik, değiştirilmiş veya hatalı şirket "
    "isimlerini tespit eden yerel eşleştirme sistemi."
)


with st.sidebar:
    st.header("Sistem")

    backend_url = st.text_input(
        "Backend adresi",
        value="http://127.0.0.1:8000",
    )

    refresh_button = st.button(
        "Sistem durumunu yenile",
        use_container_width=True,
    )


st.subheader("Sistem durumu")

try:
    health = get_backend_health(backend_url=backend_url)

except BackendConnectionError as exc:
    st.error(str(exc))
    st.info(
        "Backend'i başlatmak için proje ana dizininde şu komutu çalıştırın:\n\n"
        "`python -m uvicorn app.main:app --app-dir backend "
        "--host 127.0.0.1 --port 8000 --reload`"
    )
    st.stop()


gpu = health["gpu"]
components = health["components"]

backend_column, cuda_column, model_column = st.columns(3)

with backend_column:
    st.metric(
        label="Backend",
        value="Çalışıyor" if health["status"] == "healthy" else "Hatalı",
    )

with cuda_column:
    st.metric(
        label="CUDA",
        value="Aktif" if gpu["cuda_available"] else "Pasif",
    )

with model_column:
    st.metric(
        label="Embedding modeli",
        value=components["embedding_model"],
    )


st.divider()

st.subheader("GPU bilgileri")

if gpu["cuda_available"]:
    first_column, second_column, third_column = st.columns(3)

    with first_column:
        st.write("**Ekran kartı**")
        st.write(gpu["gpu_name"])

        st.write("**PyTorch sürümü**")
        st.write(gpu["torch_version"])

    with second_column:
        st.write("**CUDA sürümü**")
        st.write(gpu["cuda_version"])

        st.write("**cuDNN sürümü**")
        st.write(gpu["cudnn_version"])

    with third_column:
        st.write("**Toplam VRAM**")
        st.write(f'{gpu["total_memory_gb"]} GB')

        st.write("**Compute Capability**")
        st.write(gpu["compute_capability"])

    memory_column_1, memory_column_2 = st.columns(2)

    with memory_column_1:
        st.metric(
            "Kullanılan VRAM",
            f'{gpu["allocated_memory_gb"]} GB',
        )

    with memory_column_2:
        st.metric(
            "Rezerve VRAM",
            f'{gpu["reserved_memory_gb"]} GB',
        )

else:
    st.warning(
        "PyTorch CUDA ekran kartını kullanamıyor. "
        "Sistem şu anda CPU modunda çalışıyor."
    )


if gpu.get("error"):
    st.error(gpu["error"])


with st.expander("Backend tarafından dönen ham JSON"):
    st.json(health)


st.divider()

st.subheader("Proje bileşenleri")

component_labels = {
    "backend": "FastAPI backend",
    "data_source": "Aktif veri kaynağı",
    "database": "PostgreSQL bağlantısı",
    "database_details": (
        "PostgreSQL ayrıntıları"
    ),
    "company_index": (
        "Şirket arama indeksi"
    ),
    "company_index_stats": (
        "İndeks istatistikleri"
    ),
    "embedding_model": (
        "Embedding modeli"
    ),
    "embedding_model_details": (
        "Embedding model ayrıntıları"
    ),
    "reranker_model": (
        "Reranker modeli"
    ),
    "reranker_model_details": (
        "Reranker model ayrıntıları"
    ),
    "catboost_model": (
        "CatBoost modeli"
    ),
}

for component_name, component_status in components.items():
    label = component_labels.get(
        component_name,
        component_name,
    )

    if isinstance(component_status, dict):
        st.write(f"**{label}**")
        st.json(component_status)

    elif component_status == "ready":
        st.success(f"{label}: hazır")

    else:
        st.info(f"{label}: {component_status}")