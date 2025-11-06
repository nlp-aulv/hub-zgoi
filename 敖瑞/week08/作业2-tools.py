from pydantic import BaseModel, Field
from typing import List, Dict
from typing_extensions import Literal
import openai


client = openai.OpenAI(
    api_key='sk-5070859462a14565a7d30fa7778267b2',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


with open('domains.txt', 'r', encoding='utf8') as f:
    domains = [i.replace('\n', '') for i in f.readlines()]

with open('intents.txt', 'r', encoding='utf8') as f:
    intents = [i.replace('\n', '') for i in f.readlines()]

with open('slots.txt', 'r', encoding='utf8') as f:
    slots = [i.replace('\n', '') for i in f.readlines()]


DomainType = Literal[tuple(domains)]
IntentsType = Literal[tuple(intents)]
SlotsKeyType = Literal[tuple(slots)]



class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                'role': 'user',
                'content': user_prompt
            }
        ]

        tools = [
            {
                'type': 'function',
                'function': {
                    'name': response_model.model_json_schema()['title'],
                    'description': response_model.model_json_schema()['description'],
                    'parameters': {
                        'type': 'object',
                        'properties': response_model.model_json_schema()['properties'],
                        'required': response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice='auto',
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class Text(BaseModel):
    """抽取实体"""
    domain: DomainType = Field(description='领域')
    intent: IntentsType = Field(description='意图')
    slots: Dict[SlotsKeyType, str] = Field(description='实体标签字典，键为实体类型，值为实体值', default_factory=dict)


def tools_ner(query):
    result = ExtractionAgent(model_name='qwen-plus').call(query, Text)
    return result
