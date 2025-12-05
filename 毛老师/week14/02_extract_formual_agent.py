import base64
import glob
import json
import os
from io import BytesIO

import dashscope
from dashscope import Generation

import pdfplumber

# https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=2860683
api_key = "sk-c4395731abd4446b8642c7734c8dbf56"

CONVERT_TOOL_PROMPT = """
理解上述内容，并将上述内容转换为一个建模工具，需要输出信息为tool的定义，输出为json格式：
{
    "type": "function",
    "name": "建模对应的函数名字",
    "description": "建模背景的介绍",
    "parameters": {
        "type": "object",
        "properties": {
            "需要传入的参数1": {
                "type": "类型",
                "description": "参数说明",
            },
        },
        "required": ["参数1"],
    },
    "sympy": "使用传入参数使用sympy计算的完整代码，不要写注释，从symbols定义开始。"
    "return": "返回格式和参数说明"
}
"""

WRITE_CODE_PROMPT = """
理解上述函数定义，并将上述内容转换为mcp工具代码，格式如下。要写清楚函数的定义、参数说明和函数参数。
```python
import math # 必须保留
import numpy as np # 必须保留
from fastmcp import FastMCP # 必须保留
from typing import Annotated, Union # 必须保留
mcp = FastMCP(name="") # 必须保留

@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    '''Retrieves the current weather data using the city's Pinyin name.'''
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}", timeout=5).json()["data"]
    except:
        return []
```
"""



def parse_image_by_qwen_ocr(base64_image: str) -> str:
    """
    通过qwen-vl-ocr 解析图
    :param base64_image: 图的base64编码
    :return: 识别结果
    """
    try:
        messages = [{
            "role": "user",
            "content": [
                {
                    "image": f"data:image/png;base64,{base64_image}",
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
            ocr_options={"task": "document_parsing"}
        )
        return response["output"]["choices"][0]["message"].content[0]["text"]
    except:
        print("parse image using qwen error!")
        return ""


def parse_document_file(path: str):
    """
    读取原始的文档，并进行内容转换
    :param path: 文档路径
    :return:
    """
    if path.endswith(".md"):
        lines = open(path).readlines()
        return "\n".join(lines)

    images_content = []
    if path.endswith(".pdf"):
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                im = page.to_image(resolution=150)
                img_buffer = BytesIO()
                im.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                images_content.append(parse_image_by_qwen_ocr(img_base64))

    return "\n".join(images_content)


for path in glob.glob("documents/*")[:20]:
    if not os.path.exists("json"):
        os.mkdir("json")

    if not os.path.exists("code"):
        os.mkdir("code")

    print(path)
    content = parse_document_file(path)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': content + "\n" + CONVERT_TOOL_PROMPT}
    ]
    response = Generation.call(
        api_key=api_key,
        model="qwen-plus",
        messages=messages,
        result_format="message"
    )
    tool_json = response.output.choices[0].message.content
    tool_json = tool_json.strip("```json").strip("```")

    with open("./json/" + os.path.basename(path).split(".")[0] + ".json", "w") as up:
        up.writelines(tool_json)

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': tool_json + "\n" + WRITE_CODE_PROMPT}
    ]
    response = Generation.call(
        api_key=api_key,
        model="qwen-plus",
        messages=messages,
        result_format="message"
    )
    py_code = response.output.choices[0].message.content
    py_code = py_code.strip("```python").strip("```")
    print(py_code)

    with open("./code/" + os.path.basename(path).split(".")[0] + ".py", "w") as up:
        up.writelines(py_code)