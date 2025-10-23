import os
import re
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import Dict, Optional, List

# 设置环境变量
os.environ["OPENAI_API_KEY"] = "sk-3ee7446419d8497a8050245c22455530"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 初始化模型
model = init_chat_model("qwen-plus", model_provider="openai")

# 定义提示模板
system_template = "对下面的文本意图识别，类型有（QUERY，SEND，TRANSLATION），领域识别，类型有（music，message，translation），提取实体"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)


# 意图识别响应模型
class IntentDetectionResponse(BaseModel):
    """
    意图识别响应模型
    """
    text: str
    intent: str
    domain: str
    entities: Dict[str, str]
    raw_response: str


def parse_response(content: str) -> Dict[str, str]:
    """
    解析模型返回的内容，提取意图、领域和实体

    Args:
        content: 模型返回的文本内容

    Returns:
        Dict: 包含解析后的字段
    """
    result = {
        "intent": "",
        "domain": "",
        "entities": {}
    }

    try:
        # 提取意图
        intent_match = re.search(r"意图识别[：:]?\s*([A-Z]+)", content)
        if intent_match:
            result["intent"] = intent_match.group(1).strip()

        # 提取领域
        domain_match = re.search(r"领域识别[：:]?\s*([a-z]+)", content)
        if domain_match:
            result["domain"] = domain_match.group(1).strip()

        # 提取实体
        entities_match = re.search(r"实体提取[：:]?(.+)", content)
        if entities_match:
            entities_text = entities_match.group(1).strip()
            # 解析实体，如"庄小雷（接收人）"
            entity_pattern = r"([^（]+)（([^）]+)）"
            entities = re.findall(entity_pattern, entities_text)
            for entity, entity_type in entities:
                result["entities"][entity_type] = entity.strip()

        return result
    except Exception as e:
        print(f"解析响应时出错: {e}")
        return result


def detect_intent(text: str) -> IntentDetectionResponse:
    """
    对输入文本进行意图识别

    Args:
        text: 需要识别的文本

    Returns:
        IntentDetectionResponse: 包含意图、领域、实体等信息的响应对象
    """
    try:
        # 构建提示
        prompt = prompt_template.invoke({"text": text})

        # 调用模型
        response = model.invoke(prompt)

        # 解析响应内容
        parsed_result = parse_response(response.content)

        return IntentDetectionResponse(
            text=text,
            intent=parsed_result["intent"],
            domain=parsed_result["domain"],
            entities=parsed_result["entities"],
            raw_response=response.content
        )
    except Exception as e:
        raise ValueError(f"意图识别失败: {str(e)}")


# 测试函数
def test_detection():
    """
    测试意图识别功能
    """
    test_text = "给庄小雷发短信"
    result = detect_intent(test_text)
    print(f"输入文本: {test_text}")
    print(f"识别结果: {result.dict()}")


if __name__ == "__main__":
    test_detection()