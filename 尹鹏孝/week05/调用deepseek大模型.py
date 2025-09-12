import openai

client = openai.OpenAI(
    api_key="sk-57f5c13223084*****92f7d1b918a",
    base_url="https://api.deepseek.com"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "说一说尹鹏孝这个人"},
        {"role": "user", "content": "Hello"},
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
