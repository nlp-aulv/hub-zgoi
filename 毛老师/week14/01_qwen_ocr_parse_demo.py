import os
import dashscope

# https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2860683
api_key = "sk-c4395731abd4446b8642c7734c8dbf56"

messages = [{
    "role": "user",
    "content": [
        {
            "image": "file://" + os.path.abspath("./demo.png"),
            "min_pixels": 32 * 32 * 3,
            "max_pixels": 32 * 32 * 8192,
            "enable_rotate": False
        }
    ]
}]

response = dashscope.MultiModalConversation.call(
    api_key=api_key,
    model='qwen-vl-ocr-latest',
    messages=messages,
    ocr_options={"task": "advanced_recognition"}
)
print("\n\n\n高精度识别：")
print(response["output"]["choices"][0]["message"].content[0]["text"])


response = dashscope.MultiModalConversation.call(
    api_key=api_key,
    model='qwen-vl-ocr-latest',
    messages=messages,
    ocr_options={"task": "document_parsing"}
)
print("\n\n\n文档解析：")
print(response["output"]["choices"][0]["message"].content[0]["text"])

