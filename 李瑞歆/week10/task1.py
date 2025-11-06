import dashscope
from dashscope import MultiModalConversation

# 设置API Key
dashscope.api_key = "sk-e9961a40093b46d98ca6cf4be6fffc11"  # 替换为您的API Key

# 准备消息
messages = [
    {
        "role": "user",
        "content": [
            {
                "image": "./pic/pic1.jpg"
            },
            {
                "text": "识别图片的动物是猫还是狗"
            }
        ]
    }
]

# 调用云端模型
response = MultiModalConversation.call(
    model="qwen-vl-plus",  # 或 "qwen-vl-max"
    messages=messages,
    max_tokens=128
)

# 输出结果
print(response.output.choices[0].message.content)

# [{'text': '这张图片中的动物是一只狼，而不是猫或狗。以下是一些识别狼的关键特征：\n\n
# 1. **体型和外观**：狼的体型较大，肌肉发达，毛发浓密且长。图片中的动物有明显的狼的特征，如强壮的四肢和粗壮的尾巴。\n\n
# 2. **面部特征**：狼的面部较长，耳朵直立且尖锐，眼睛呈黄色或琥珀色。图片中的动物有这些特征，尤其是耳朵和眼睛。\n\n
# 3. **毛色**：狼的毛色通常为灰、棕、白相间，图片中的动物毛色'}]