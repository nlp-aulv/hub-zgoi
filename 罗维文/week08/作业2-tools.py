import openai
import json
from pydantic import BaseModel, Field # 定义传入的数据请求格式
from typing import List, Optional, Dict, Any
from typing_extensions import Literal

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-78cc4e9ac8f44efdb207b7232e1ae6d8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 领域类别
with open('E:/AI学习/02-week08-joint-bert-training-only/domains.txt', 'r', encoding='utf-8') as fp:
    domains = fp.read().split('\n')
    domains = Literal[tuple(domains)]

# 意图类别
with open('E:/AI学习/02-week08-joint-bert-training-only/intents.txt', 'r', encoding='utf-8') as fp:
    intents = fp.read().split('\n')
    intents = Literal[tuple(intents)]

# 实体标签
with open('E:/AI学习/02-week08-joint-bert-training-only/slots.txt', 'r', encoding='utf-8') as fp:
    slots = fp.read().split('\n')
    # slots = Literal[tuple(slots)]
    # print(slots)
    slots = ' / '.join(slots)

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
                        "required": response_model.model_json_schema()['required'], # 必须要传的参数
                    },
                }
            }
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            # tool_choice={"type": "function", "function": {"name": response_model.model_json_schema()['title']}}
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            # print('   ====    ', response.choices[0].message)
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print('ERROR', response.choices[0].message, e)
            return None

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: domains = Field(description="领域")
    intent: intents = Field(description="意图")
    # slots: Dict[slots,  List[str]] = Field(description=f"实体")
    slots: Dict[str,  List[str]] = Field(description=f"实体。实体类型必须是以下之一:{slots}")

result = ExtractionAgent(model_name = "qwen-plus").call("帮我查询下从北京到天津到武汉的汽车票", IntentDomainNerTask)
print(result)

with open('E:/AI学习/02-week07-joint-bert-training-only/data/test.json', 'r', encoding='utf-8') as fp:
    pred_data = eval(fp.read())
    for i, p_data in enumerate(pred_data):
        text = p_data['text']
        print('=================================')
        print(text)
        result = ExtractionAgent(model_name="qwen-plus").call(text, IntentDomainNerTask)
        print('=================================')
        print(result)
        if i == 10:
            break
