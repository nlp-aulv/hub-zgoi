from openai import OpenAI

client = OpenAI(
    api_key="sk-a68466c1affa4721b9e07e0314b495dd", 
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "解释下rag主要的流程"},
    ],
    stream=False
)

print(response.choices[0].message.content)
