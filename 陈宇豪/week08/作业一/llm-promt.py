import os

from langchain.chains import LLMChain
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from promt import DOMAIN_PROMPT

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


# prompt = prompt_template.invoke(
#     {"domain": process_file("domains.txt"), "intents": process_file("intents.txt"),
#      "entity": process_file("slots.txt")})
# print("提示词为:", prompt.to_messages())

def chat_model_domain(message: str):
    prompt = prompt_template.invoke(
        {"domain": process_file("domains.txt"), "intents": process_file("intents.txt"),
         "entity": process_file("slots.txt"), "text": message})
    result = chat_model.invoke(prompt)
    print("语句:\"{}\"的分词结果:{}".format(message, result.content))


with open("sentences.txt", "r", encoding="utf-8") as f:
    chat_model_domain("张培的电话号码是多少")
