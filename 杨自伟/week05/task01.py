import openai

client = openai.OpenAI(
    api_key="sk-166bfb70f3624af6992e33192138e8ef",
    base_url="https://api.deepseek.com"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "deepseek介绍"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
