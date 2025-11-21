import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
    api_key="sk-88e148349f094d10b62f6faf49556031",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

completion = client.chat.completions.create(
    # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
    model="qwen3-vl-flash",
    messages = [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
        {"type": "text", "text": "just tell me dog or cat? Without any other words"},
    ]}]
)
print(completion.choices[0].message.content)
