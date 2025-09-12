import openai

client = openai.OpenAI(
    api_key="sk-56b3fb3db*****8b8efab74792f3",
    base_url="https://api.deepseek.com"
)

completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "你是一个帮助小助手，准确回答用户提出的问题"},
        {"role": "user", "content": "你是谁"}
    ],
)

print(completion.choices[0].message.content)
