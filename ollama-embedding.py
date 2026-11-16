from src.ollama_client import get_ollama_embeddings


if __name__ == "__main__":
    for embedding in get_ollama_embeddings(
        ["这是要转换为嵌入向量的文本", "文本2", "文本3"]
    ):
        print(embedding)
