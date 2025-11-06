
from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import Literal as LiteralExt

import openai

client = openai.OpenAI(
    api_key="sk-4806ae58c8de41848fd1153108c3d86c",  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 读取领域类别文件
with open('domains.txt', 'r', encoding='utf-8') as f:
    domains = [line.strip() for line in f.readlines()]

# 读取意图类别文件
with open('intents.txt', 'r', encoding='utf-8') as f:
    intents = [line.strip() for line in f.readlines()]

# 读取实体类别文件
with open('slots.txt', 'r', encoding='utf-8') as f:
    slots = [line.strip() for line in f.readlines()]

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
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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

# 领域识别模型
class DomainClassifier(BaseModel):
    """识别用户输入文本所属的领域"""
    domain: str = Field(description="用户输入文本所属的领域类别")

# 意图识别模型
class IntentClassifier(BaseModel):
    """识别用户输入文本的意图"""
    intent: str = Field(description="用户输入文本的意图类别")

# 实体识别模型
class SlotClassifier(BaseModel):
    """识别用户输入文本中的实体"""
    slots: List[str] = Field(description="用户输入文本中包含的实体类别列表")

# 领域识别函数
def classify_domain(text: str, model_name: str = "qwen-plus"):
    """对用户输入文本进行领域分类"""
    agent = ExtractionAgent(model_name)
    result = agent.call(text, DomainClassifier)
    return result.domain if result else None

# 意图识别函数
def classify_intent(text: str, model_name: str = "qwen-plus"):
    """对用户输入文本进行意图分类"""
    agent = ExtractionAgent(model_name)
    result = agent.call(text, IntentClassifier)
    return result.intent if result else None

# 实体识别函数
def classify_slots(text: str, model_name: str = "qwen-plus"):
    """对用户输入文本进行实体识别"""
    agent = ExtractionAgent(model_name)
    result = agent.call(text, SlotClassifier)
    return result.slots if result else None

# 测试示例
if __name__ == "__main__":
    test_text = "我的大哥想听周杰伦和薛之谦的歌"

    print("领域识别结果:")
    domain_result = classify_domain(test_text)
    print(domain_result)

    print("\n意图识别结果:")
    intent_result = classify_intent(test_text)
    print(intent_result)

    print("\n实体识别结果:")
    slot_result = classify_slots(test_text)
    print(slot_result)
