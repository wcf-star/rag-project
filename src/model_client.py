# -*- coding: utf-8 -*-
# 统一的模型 / 嵌入客户端。
# 通过 .env 中的 LLM_PROVIDER / EMBEDDING_PROVIDER 在「本地 Ollama」与
# 「云端千问 DashScope」之间切换，两套 provider 都走 OpenAI 兼容接口。
import os
from typing import List, Optional, Union

import requests
from dotenv import load_dotenv
from openai import OpenAI


# ---------------------------------------------------------------------------
# provider 解析
# ---------------------------------------------------------------------------

def get_llm_provider() -> str:
    """当前生成模型 provider：ollama | qwen"""
    load_dotenv()
    return os.getenv("LLM_PROVIDER", "ollama").strip().lower()


def get_embedding_provider() -> str:
    """当前嵌入模型 provider：ollama | qwen。

    默认跟随 LLM_PROVIDER，可通过 EMBEDDING_PROVIDER 单独覆盖，
    例如「千问生成 + 本地 bge-m3 嵌入」或反之。
    """
    load_dotenv()
    return os.getenv(
        "EMBEDDING_PROVIDER",
        os.getenv("LLM_PROVIDER", "ollama"),
    ).strip().lower()


# ---------------------------------------------------------------------------
# Ollama 本地配置
# ---------------------------------------------------------------------------

def get_ollama_base_url() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_ollama_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def get_ollama_embedding_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_EMBEDDING", "bge-m3")


# ---------------------------------------------------------------------------
# 千问 DashScope 云端配置（OpenAI 兼容模式）
# ---------------------------------------------------------------------------

def get_qwen_api_key() -> str:
    load_dotenv()
    key = os.getenv("QWEN_API_KEY", os.getenv("DASHSCOPE_API_KEY", "")).strip()
    if not key:
        raise ValueError("未设置 QWEN_API_KEY（或 DASHSCOPE_API_KEY），请在 .env 中配置。")
    return key


def get_qwen_base_url() -> str:
    load_dotenv()
    return os.getenv(
        "QWEN_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).rstrip("/")


def get_qwen_model() -> str:
    load_dotenv()
    return os.getenv("QWEN_MODEL", "qwen-plus")


def get_qwen_embedding_model() -> str:
    load_dotenv()
    return os.getenv("QWEN_EMBEDDING", "text-embedding-v3")


def get_qwen_embedding_dimensions() -> int:
    load_dotenv()
    # text-embedding-v3/v4 支持 1024/768/512 等；设为 0 表示不传该参数
    return int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "1024"))


# ---------------------------------------------------------------------------
# 当前默认模型（随 provider 切换）
# ---------------------------------------------------------------------------

def get_default_model() -> str:
    provider = get_llm_provider()
    if provider == "qwen":
        return get_qwen_model()
    return get_ollama_model()


def get_default_embedding_model() -> str:
    provider = get_embedding_provider()
    if provider == "qwen":
        return get_qwen_embedding_model()
    return get_ollama_embedding_model()


# ---------------------------------------------------------------------------
# 客户端构造
# ---------------------------------------------------------------------------

def create_llm_client(provider: Optional[str] = None) -> OpenAI:
    """按 provider 返回 OpenAI 兼容客户端。"""
    provider = (provider or get_llm_provider()).lower()
    if provider == "qwen":
        return OpenAI(
            api_key=get_qwen_api_key(),
            base_url=get_qwen_base_url(),
            timeout=300,
            max_retries=3,
        )
    if provider == "ollama":
        return create_ollama_client()
    raise ValueError(f"不支持的 LLM provider: {provider}")


def create_ollama_client() -> OpenAI:
    return OpenAI(
        api_key="ollama",
        base_url=f"{get_ollama_base_url()}/v1",
        timeout=None,
        max_retries=2,
    )


# ---------------------------------------------------------------------------
# 嵌入向量
# ---------------------------------------------------------------------------

def get_ollama_embeddings(
    texts: Union[str, List[str]], model: Optional[str] = None
) -> List[List[float]]:
    """通过 Ollama 本地 /api/embed 接口获取嵌入。"""
    load_dotenv()
    inputs = [texts] if isinstance(texts, str) else texts
    if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
        raise ValueError("Embedding inputs must be non-empty strings.")

    response = requests.post(
        f"{get_ollama_base_url()}/api/embed",
        json={"model": model or get_ollama_embedding_model(), "input": inputs},
        timeout=120,
    )
    response.raise_for_status()
    payload = response.json()
    embeddings = payload.get("embeddings")
    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(inputs)
        or any(not isinstance(embedding, list) or not embedding for embedding in embeddings)
    ):
        raise RuntimeError(f"Unexpected Ollama embedding response: {payload}")
    return embeddings


def get_qwen_embeddings(
    texts: Union[str, List[str]], model: Optional[str] = None
) -> List[List[float]]:
    """通过千问 DashScope OpenAI 兼容 /embeddings 接口获取嵌入。"""
    load_dotenv()
    inputs = [texts] if isinstance(texts, str) else texts
    if not inputs or any(not isinstance(text, str) or not text.strip() for text in inputs):
        raise ValueError("Embedding inputs must be non-empty strings.")

    model = model or get_qwen_embedding_model()
    client = OpenAI(
        api_key=get_qwen_api_key(),
        base_url=get_qwen_base_url(),
        timeout=180,
        max_retries=3,
    )

    # 千问 embedding 接口单次请求有批量上限，这里按 10 条分批
    batch_size = 10
    embeddings: List[List[float]] = []
    for i in range(0, len(inputs), batch_size):
        batch = inputs[i:i + batch_size]
        kwargs = {}
        if "v3" in model or "v4" in model:
            dims = get_qwen_embedding_dimensions()
            if dims:
                kwargs["dimensions"] = dims
        resp = client.embeddings.create(model=model, input=batch, **kwargs)
        data = sorted(resp.data, key=lambda d: getattr(d, "index", 0))
        embeddings.extend([d.embedding for d in data])
    return embeddings


def get_embeddings(
    texts: Union[str, List[str]],
    model: Optional[str] = None,
    provider: Optional[str] = None,
) -> List[List[float]]:
    """统一的嵌入入口，按 provider 分发。"""
    provider = (provider or get_embedding_provider()).lower()
    if provider == "qwen":
        return get_qwen_embeddings(texts, model=model)
    if provider == "ollama":
        return get_ollama_embeddings(texts, model=model)
    raise ValueError(f"不支持的 embedding provider: {provider}")
