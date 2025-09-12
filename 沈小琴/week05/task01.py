from openai import OpenAI

client = OpenAI(api_key="sk-50f13f999******6b5c36f9707e", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "描述RAG的实现流程，写为文档"},
    ],
    stream=False
)

print(response.choices[0].message.content)
