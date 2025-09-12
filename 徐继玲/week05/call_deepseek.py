import openai

client = openai.OpenAI(
    api_key="sk-56b3fb3*******28b8efab74792f3",
    base_url="https://api.deepseek.com"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个帮助小助手"},
        {"role": "user", "content": "今天的天气怎么样"}
    ],
)
print(completion.choices[0].message.content)
