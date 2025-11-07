from dashscope import MultiModalConversation  # 多模态对话功能
import dashscope  # DashScope SDK主库
import base64

dashscope.api_key = "sk-cc43ca2821f64bfa9c6e20bf0889d92c"


def classify_image(image_path):
    """
    图片分类
    """
    try:
        #读取图片并编码
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        #调用大模型
        response = MultiModalConversation.call(
            model="qwen-vl-plus",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "image": f"data:image/jpeg;base64,{image_data}"
                        },
                        {
                            "text": "请仔细识别这张图片，回答图片中是猫、狗，还是两者都不是？请只回答'猫'、'狗'或'都不是'"
                        }
                    ]
                }
            ]
        )

        return response.output.choices[0].message.content

    except Exception as e:
        print(f"Error: {e}")
        return "无法识别图片"


print(classify_image("img01.jpg"))

#输出：[{'text': '狗'}]
