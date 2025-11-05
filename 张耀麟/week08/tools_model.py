from fastapi import APIRouter, Query
from utils import LLM

tools = APIRouter()

@tools.get("/", description="这是使用tools进行意图识别+领域识别+实体识别的接口,返回json")
def predict(text: str = Query(default=None, description="输入句子用于全识别")):
    llm = LLM("deepseek-chat")
    response = llm.call(text)
    if response is None:
        print("Query Failed")
        return {}
    return response


@tools.get("/intent", description="这是使用tools进行意图识别的接口")
def intent_recognization(text: str = Query(default=None, description="输入句子用于意图识别")):
    response = predict(text)
    return response.get("intent", "Query Failed")


@tools.get("/field", description="这是使用tools进行领域识别的接口")
def field_recognization(text: str = Query(default=None, description="输入句子用于领域识别")):
    response = predict(text)
    return response.get("domain", "Query Failed")


@tools.get("/ner", description="这是使用tools进行实体识别的接口")
def ner_recognization(text: str = Query(default=None, description="输入句子用于实体识别")):
    response = predict(text)
    return response.get("slots", "Query Failed")
