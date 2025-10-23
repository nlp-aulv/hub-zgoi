# -*- coding: utf-8 -*-
import openai
import json
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional
from typing_extensions import Literal
from dotenv import load_dotenv
import os
# 加载环境变量（密钥放在.env文件中）
load_dotenv()
# 初始化客户端（
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY', "sk-411bf89559914810893fd40f59a24515"),
    base_url= os.getenv('OPENAI_BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)


#加载进来的数据格式化成一个数组
def file_exchange(file_path: str) -> List[str]:
    """字符串列表"""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    # 去除空行和空白
    items = [line.strip() for line in content.split('\n') if line.strip()]
    return items
"""
这个智能体（不是满足agent的功能），能自动生成tools的json，实现信息信息抽取
"""
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'], # 工具名字
                    "description": response_model.model_json_schema()['description'], # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'], # 参数说明
                        # "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

 # 从文件读取数据
domain_data = file_exchange('domains.txt')
intent_data = file_exchange('intents.txt')

# 动态创建 Literal 类型（如果数据不为空）
if domain_data:
    DomainLiteral = Literal[tuple(domain_data)]
else:
    DomainLiteral = str  # 备用类型

if intent_data:
    IntentLiteral = Literal[tuple(intent_data)]
else:
    IntentLiteral = str  # 备用类型
class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""

    domain:DomainLiteral = Field(description="领域")
    intent: IntentLiteral= Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: List[str] = Field(description="目的地")


def llm_tools(query='帮我查询下从北京到天津到武汉的汽车票'):
    print('输出query-：',query)
    result = ExtractionAgent(model_name = "qwen-plus").call(query, IntentDomainNerTask)
    print(result)
    dict_output = result.model_dump()
    pretty_json = json.dumps(dict_output, indent=2, ensure_ascii=False, default=str)
    print("输出JSON")
    print(pretty_json)
    return pretty_json
# llm_tools()