from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="klm"
)

# 发送请求
response = client.chat.completions.create(
    model="qwen3:0.6b",  # 指定模型
    messages=[
        {"role": "system", "content": "你是一个心理咨询专家。"},
        {"role": "user", "content": "为什么总是会担心门没关好？"}
    ],
    temperature=0.7,  # 控制生成多样性
    max_tokens=256    # 最大生成 token 数
)

# 打印结果
print(response.choices[0].message.content)
