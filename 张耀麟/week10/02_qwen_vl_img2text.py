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
         "image_url": {"url": "https://picx.zhimg.com/v2-d89491a4e6519468a25f7c9133eba649_720w.jpg?source=172ae18b"}},
        {"type": "text", "text": "请你帮我图片转文字"},
    ]}]
)
print(completion.choices[0].message.content)
