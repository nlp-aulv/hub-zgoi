import os
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

os.environ["OPENAI_API_KEY"] = "sk-3ee7446419d8497a8050245c22455530"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

model = init_chat_model("qwen-plus", model_provider="openai")

system_template = "对下面的文本意图识别，类型有（QUERY，SEND，TRANSLATION），领域识别，类型有（music，message，translation），提取实体"
prompt_template = ChatPromptTemplate.from_messages(
    [("system", system_template), ("user", "{text}")]
)

prompt = prompt_template.invoke({"text": "给庄小雷发短信"})
print(prompt.to_messages())

response = model.invoke(prompt)
print(response.content)

"""
[SystemMessage(content='对下面的文本意图识别，类型有（QUERY，SEND，TRANSLATION），领域识别，类型有（music，message，translation），提取实体', additional_kwargs={}, response_metadata={}), HumanMessage(content='给庄小雷发短信', additional_kwargs={}, response_metadata={})]
- **意图识别**：SEND  
- **领域识别**：message  
- **实体提取**：庄小雷（接收人）
"""