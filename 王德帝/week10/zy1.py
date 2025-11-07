from dashscope import MultiModalConversation
import dashscope
from PIL import Image

dashscope.api_key = "sk-cbf9e44f6f164d2b9d4b9bbf110bbd6c"

image_path = "./zy1.jpg"

response = MultiModalConversation.call(
    model="qwen-vl-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "image": f"file://{image_path}"
                },
                {
                    "text": "请判断这张图片是一只狗还是一只猫，只回答dog或cat"
                }
            ]
        }
    ],
)

# 获取模型的回答
if response.status_code == 200:
    answer = response.output.choices[0].message.content[0]["text"]
    print("模型回答：", answer)
else:
    print("请求失败：", response.message)


(py312) dd@mac week10 % /opt/miniconda3/envs/py312/bin/python /Users/dd/Documents/nlp/projects/week10/zy1.py
模型回答： cat
