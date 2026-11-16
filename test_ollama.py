from src.ollama_client import create_ollama_client, get_ollama_model


review = "这款音效特别好 给你意想不到的音质。"
client = create_ollama_client()
response = client.chat.completions.create(
    model=get_ollama_model(),
    messages=[
        {
            "role": "system",
            "content": "你是一名舆情分析师，帮我判断产品口碑的正负向，回复请用一个词语：正向 或者 负向",
        },
        {"role": "user", "content": review},
    ],
)
print(response.choices[0].message.content)
