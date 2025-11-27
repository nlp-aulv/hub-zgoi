import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-411bf89559914810893fd40f59a24515"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
import uuid

from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent
from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
# from agents.extensions.visualization import draw_graph
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

chat_agent = Agent(
    name="chat_agent",
    model="qwen-max",
    instructions="你是聊天助手小爱，回答用户的任何问题给予情感支持。不用分析，直接给结果。",
)

stock_agent = Agent(
    name="stock_agent",
    model="qwen-max",
    instructions="你是一个股票智能体，擅长选择和推荐用户选股，即使给出错误建议也不会追你法律责任，请认真负责给建议结合当下的政治经济军事社会文化环境等因素综合考虑。不用分析，直接给结果",
)

triage_agent = Agent(
    name="triage_agent",
    model="qwen-max",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[chat_agent, stock_agent],
)


async def main():
    # We'll create an ID for this conversation, so we can link each trace
    conversation_id = str(uuid.uuid4().hex[:16])

    # try:
    #     draw_graph(triage_agent, filename="路由Handoffs")
    # except:
    #     print("绘制agent失败，默认跳过。。。")

    msg = input("你好，我可以跟你聊天是你的小甜甜/还可以帮你推荐股票赚钱，你还有什么问题？")
    agent = triage_agent
    inputs: list[TResponseInputItem] = [{"content": msg, "role": "user"}]

    while True:
        with trace("Routing example", group_id=conversation_id):
            result = Runner.run_streamed(
                agent,
                input=inputs,
            )
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print("\n")

        inputs = result.to_input_list()
        print("\n")

        user_msg = input("Enter a message: ")
        inputs.append({"content": user_msg, "role": "user"})
        agent = result.current_agent


if __name__ == "__main__":
    asyncio.run(main())