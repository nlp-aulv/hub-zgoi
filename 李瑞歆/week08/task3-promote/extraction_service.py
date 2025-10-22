from pydantic import BaseModel, Field
from typing import List, Optional
from typing_extensions import Literal
import openai

client = openai.OpenAI(
    api_key="sk-3ee7446419d8497a8050245c22455530",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt: str, response_model):
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

        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(f'ERROR: {e}')
            return None


# 意图识别模型
class IntentRecognitionRequest(BaseModel):
    """意图识别请求模型"""
    text: str = Field(description="需要识别意图的文本")


class IntentRecognitionResponse(BaseModel):
    """意图识别响应模型"""
    domain: Literal["商品查询", "订单管理", "物流跟踪", "售后服务", "支付问题", "账户管理", "促销活动"] = Field(
        description="业务领域")
    intent: Literal["搜索", "比价", "下单", "退换货", "咨询", "投诉", "催单"] = Field(description="具体意图")
    product_category: List[str] = Field(description="涉及的商品类别")


# 领域识别模型
class DomainRecognitionRequest(BaseModel):
    """领域识别请求模型"""
    text: str = Field(description="需要识别领域的文本")


class DomainRecognitionResponse(BaseModel):
    """领域识别响应模型"""
    discipline: Literal[
        "计算机科学", "物理学", "化学", "生物学", "医学", "工程学", "数学", "经济学", "心理学", "文学", "历史学", "哲学"] = Field(
        description="学科门类")
    subfield: str = Field(description="具体研究子领域")
    research_type: Literal["理论研究", "实验研究", "应用研究", "综述", "案例分析"] = Field(description="研究类型")


# 实体识别模型
class EntityRecognitionRequest(BaseModel):
    """实体识别请求模型"""
    text: str = Field(description="需要识别实体的文本")


class EntityRecognitionResponse(BaseModel):
    """实体识别响应模型"""
    persons: List[str] = Field(description="人名")
    organizations: List[str] = Field(description="组织机构名")
    locations: List[str] = Field(description="地名")
    time_expressions: List[str] = Field(description="时间表达式")
    other_entities: List[str] = Field(description="其他重要实体")


class ExtractionService:
    def __init__(self):
        self.agent = ExtractionAgent()

    def recognize_intent(self, text: str) -> Optional[IntentRecognitionResponse]:
        """意图识别"""

        class Text(BaseModel):
            """意图识别：识别电商场景下的用户意图"""
            domain: Literal["商品查询", "订单管理", "物流跟踪", "售后服务", "支付问题", "账户管理", "促销活动"] = Field(
                description="业务领域")
            intent: Literal["搜索", "比价", "下单", "退换货", "咨询", "投诉", "催单"] = Field(description="具体意图")
            product_category: List[str] = Field(description="涉及的商品类别")

        return self.agent.call(text, Text)

    def recognize_domain(self, text: str) -> Optional[DomainRecognitionResponse]:
        """领域识别"""

        class Text(BaseModel):
            """领域识别：学术研究领域分类"""
            discipline: Literal[
                "计算机科学", "物理学", "化学", "生物学", "医学", "工程学", "数学", "经济学", "心理学", "文学", "历史学", "哲学"] = Field(
                description="学科门类")
            subfield: str = Field(description="具体研究子领域")
            research_type: Literal["理论研究", "实验研究", "应用研究", "综述", "案例分析"] = Field(
                description="研究类型")

        return self.agent.call(text, Text)

    def recognize_entities(self, text: str) -> Optional[EntityRecognitionResponse]:
        """实体识别"""

        class Text(BaseModel):
            """通用实体识别"""
            persons: List[str] = Field(description="人名")
            organizations: List[str] = Field(description="组织机构名")
            locations: List[str] = Field(description="地名")
            time_expressions: List[str] = Field(description="时间表达式")
            other_entities: List[str] = Field(description="其他重要实体")

        return self.agent.call(text, Text)


# 创建全局服务实例
extraction_service = ExtractionService()