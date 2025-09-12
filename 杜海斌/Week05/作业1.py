from openai import OpenAI

client = OpenAI(
    api_key="sk-4ddd203e863d42f3ab8b1bc765ffce20",  # 替换为实际密钥
    base_url="https://api.deepseek.com"  # DeepSeek API地址
)

for resp in client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "解释卷积神经网络（CNN）的工作原理。"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
