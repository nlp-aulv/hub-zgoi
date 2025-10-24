import json
import os

from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool

from promt import DOMAIN_PROMPT
from Domain import Response
from 陈宇豪.week08.作业二_tool import ExtractToll

os.environ["OPENAI_API_KEY"] = "sk-c2a0cfd8d9b14cba93ddcb0bab3e112d"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"


def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        str = list()
        for line in file.readlines():
            str.append(line.strip())
        return ",".join(str)


chat_model = init_chat_model("qwen-plus-2025-09-11", model_provider="openai")
prompt_template = ChatPromptTemplate.from_messages([("system", DOMAIN_PROMPT), ("user", "{text}")])

intents = process_file("intents.txt")
domain = process_file("domains.txt")
entity = process_file("slots.txt")

extract_toll = {
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


def chat_model_domain(message: str) -> Response:
    prompt = prompt_template.invoke(
        {"domain": domain, "intents": intents, "entity": entity, "text": message})
    chat_model_with_tools=chat_model.bind_tools([extract_toll])
    result = chat_model_with_tools.invoke(prompt)
    print("语句:\"{}\"的分词结果:\n{}".format(message, result.content))
    data = json.loads(result.content)
    # 解析JSON并转换字段名
    response_data = {
        "request_user": None,  # 因为原始JSON中没有这个字段
        "entities": data["entities"],  # 将entities映射到entity
        "domain": data["domain"],
        "intent": data["intent"],
        "error_msg": None  # 因为原始JSON中没有这个字段
    }
    return Response(**response_data)


with open("sentences.txt", "r", encoding="utf-8") as f:
    chat_model_domain("张培的电话号码是多少")
