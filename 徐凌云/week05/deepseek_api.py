from openai import OpenAI

client = OpenAI(api_key="sk-d68c0c8e1e10496db0b1c1b8b4a1d172", base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Who are you?"},
    ],
    stream=False
)

print(response.choices[0].message.content)
