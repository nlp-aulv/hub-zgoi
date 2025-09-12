import openai

client = openai.OpenAI(
    api_key="sk-1a30fafa1b7f*****fbaf46f0",
    base_url="https://api.deepseek.com/v1"
)

# 调用DeepSeek模型
for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "user", "content": "证明费马大定理"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
