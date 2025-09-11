from openai import OpenAI

# 初始化客户端，指向 Ollama 的本地服务
client = OpenAI(
    base_url="http://localhost:11434/v1",  # Ollama API 地址
    api_key=""  # Ollama 默认无需真实 API Key，填任意值即可
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个帮助小助手，准确回答用户提出的问题"},
        {"role": "user", "content": "介绍一下成为一个AI大模型工程师需要什么技能"}
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=512    # 最大生成 token 数
)

# 打印结果
print(response.choices[0].message.content)