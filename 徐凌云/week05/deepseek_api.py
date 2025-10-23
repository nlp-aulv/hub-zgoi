from openai import OpenAI

client = OpenAI(api_key="sk-d68c0***********************d172", base_url="https://api.deepseek.com/v1")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Who are you?"},
    ],
    stream=False
)

print(response.choices[0].message.content)
