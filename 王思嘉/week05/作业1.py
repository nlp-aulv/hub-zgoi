from openai import OpenAI#deepseek

client = OpenAI(api_key="sk-baeda51895ba4beb9babed8e33a83c09", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
