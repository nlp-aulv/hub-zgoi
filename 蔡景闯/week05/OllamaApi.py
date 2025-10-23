from openai import OpenAI

client = OpenAI(api_key="", base_url="http://localhost:11434/v1")

response = client.chat.completions.create(
    model="qwen3:4b",
    messages=[
        {"role": "user", "content": "讲一个冷笑话"},
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512  # 最大生成 token 数
)

print(response.choices[0].message.content)