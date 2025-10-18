import os

import openai
from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import StructuredTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing import List, Optional
from typing_extensions import Literal

from promt import DOMAIN_PROMPT

os.environ["OPENAI_API_KEY"] = "sk-c2a0cfd8d9b14cba93ddcb0bab3e112d"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-c2a0cfd8d9b14cba93ddcb0bab3e112d",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self):
        # 使用 StructuredTool 将 Pydantic 模型映射为工具
        self.tools= [
            {
                "type": "function",
                "function": {
                    "name": ExtractToll.model_json_schema()['title'],  # 工具名字
                    "description": ExtractToll.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": ExtractToll.model_json_schema()['properties'],  # 参数说明
                        # "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]

    def extract(self, text):
        messages = [
            {
                "role": "user",
                "content": text
            }
        ]
        response = client.chat.completions.create(
            model="qwen-plus-2025-09-11",
            messages=messages,
            tools=self.tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return ExtractToll.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class ExtractToll(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY'] = Field(description="意图")
    Src: Optional[str] = Field(description="出发地")
    Des: Optional[List[str]] = Field(description="目的地")
    Time: Optional[str] = Field(description="时间")
    Weather: Optional[str] = Field(description="天气")


print(ExtractionAgent().extract("张培的电话号码是多少"))
