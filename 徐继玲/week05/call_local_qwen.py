from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key=""
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个帮助小助手，准确回答用户提出的问题"},
        {"role": "user", "content": "请告诉我土耳其有什么好玩的"}
    ],
    temperature=0.7,
    max_tokens=512
)

print(response.choices[0].message.content)