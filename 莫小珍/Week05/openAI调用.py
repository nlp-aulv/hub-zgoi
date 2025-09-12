import openai

client = openai.OpenAI(
    api_key="sk-8f20fab238144f578a588040419f92a9",
    base_url="https://api.deepseek.com"
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "详细介绍大模型开发需要掌握哪些技能"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
