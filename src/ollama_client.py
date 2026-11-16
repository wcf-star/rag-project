import os
from typing import List, Optional, Union

import requests
from dotenv import load_dotenv
from openai import OpenAI


def get_llm_provider() -> str:
    load_dotenv()
    return os.getenv("LLM_PROVIDER", "ollama").lower()


def get_ollama_base_url() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")


def get_ollama_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_MODEL", "qwen2.5:7b")


def get_ollama_embedding_model() -> str:
    load_dotenv()
    return os.getenv("OLLAMA_EMBEDDING", "bge-m3")


def create_ollama_client() -> OpenAI:
    return OpenAI(
        api_key="ollama",
        base_url=f"{get_ollama_base_url()}/v1",
        timeout=None,
        max_retries=2,
    )


def get_ollama_embeddings(
    texts: Union[str, List[str]], model: Optional[str] = None
) -> List[List[float]]:
    """Create embeddings through Ollama's local /api/embed endpoint."""
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
