# -*- coding: utf-8 -*-
# 兼容旧导入路径：统一客户端已迁移到 src/model_client.py。
# 保留此文件，使 test_ollama.py / ollama-embedding.py 等旧脚本无需改动。
from src.model_client import (
    create_llm_client,
    create_ollama_client,
    get_default_embedding_model,
    get_default_model,
    get_embedding_provider,
    get_embeddings,
    get_llm_provider,
    get_ollama_base_url,
    get_ollama_embedding_model,
    get_ollama_embeddings,
    get_ollama_model,
    get_qwen_api_key,
    get_qwen_base_url,
    get_qwen_embedding_dimensions,
    get_qwen_embedding_model,
    get_qwen_embeddings,
    get_qwen_model,
)

__all__ = [
    "create_llm_client",
    "create_ollama_client",
    "get_default_embedding_model",
    "get_default_model",
    "get_embedding_provider",
    "get_embeddings",
    "get_llm_provider",
    "get_ollama_base_url",
    "get_ollama_embedding_model",
    "get_ollama_embeddings",
    "get_ollama_model",
    "get_qwen_api_key",
    "get_qwen_base_url",
    "get_qwen_embedding_dimensions",
    "get_qwen_embedding_model",
    "get_qwen_embeddings",
    "get_qwen_model",
]
