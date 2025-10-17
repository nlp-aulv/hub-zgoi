from pydantic import BaseModel, Field
from typing import List
from typing_extensions import Literal

import openai

client = openai.OpenAI(
    api_key="sk-3ee7446419d8497a8050245c22455530", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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

class Text(BaseModel):
    """意图识别：识别电商场景下的用户意图"""
    domain: Literal["商品查询", "订单管理", "物流跟踪", "售后服务", "支付问题", "账户管理", "促销活动"] = Field(description="业务领域")
    intent: Literal["搜索", "比价", "下单", "退换货", "咨询", "投诉", "催单"] = Field(description="具体意图")
    product_category: List[str] = Field(description="涉及的商品类别")

result = ExtractionAgent(model_name="qwen-plus").call('我想买一台苹果手机，最新款的，有什么优惠吗？', Text)
print(result)

class Text(BaseModel):
    """领域识别：学术研究领域分类"""
    discipline: Literal["计算机科学", "物理学", "化学", "生物学", "医学", "工程学", "数学", "经济学", "心理学", "文学", "历史学", "哲学"] = Field(description="学科门类")
    subfield: str = Field(description="具体研究子领域")
    research_type: Literal["理论研究", "实验研究", "应用研究", "综述", "案例分析"] = Field(description="研究类型")

result = ExtractionAgent(model_name="qwen-plus").call('基于深度学习的图像识别在医疗诊断中的应用研究', Text)
print(result)

class Text(BaseModel):
    """通用实体识别"""
    persons: List[str] = Field(description="人名")
    organizations: List[str] = Field(description="组织机构名")
    locations: List[str] = Field(description="地名")
    time_expressions: List[str] = Field(description="时间表达式")
    other_entities: List[str] = Field(description="其他重要实体")

result = ExtractionAgent(model_name="qwen-plus").call('昨天马云在阿里巴巴杭州总部会见了腾讯公司的马化腾，讨论了下半年的合作计划。', Text)
print(result)


"""
domain='商品查询' intent='搜索' product_category=['手机']
discipline='医学' subfield='基于深度学习的图像识别' research_type='应用研究'
persons=['马云', '马化腾'] organizations=['阿里巴巴', '腾讯公司'] locations=['杭州'] time_expressions=['昨天', '下半年'] other_entities=['合作计划']
"""