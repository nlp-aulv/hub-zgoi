from openai import OpenAI

client = OpenAI(api_key="sk-8aa396******b1ec30d6385cef4", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "讲一个冷笑话，50个字以内"},
    ],
    stream=False
)

print(response.choices[0].message.content)

# DeepSeek回答： 小明问爸爸：“寒号鸟为什么不会冻死？”爸爸答：“因为它有号（好）冷啊！”
