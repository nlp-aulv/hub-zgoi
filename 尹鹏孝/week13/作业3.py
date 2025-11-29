import asyncio
import requests  # type: ignore
from fastmcp import FastMCP, Client
import os

# https://bailian.console.aliyun.com/?tab=model#/api-key
os.environ["OPENAI_API_KEY"] = "sk-78cc4e9ac8f44efdb207b7232e1ae6d8"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from pydantic import BaseModel
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from api.autostock import app
from api.chat import mcp as chat

mcp = FastMCP.from_fastapi(app=app)
class UserQuestionOutput(BaseModel):
    """用于判断用户请求是否属于功课或学习类问题的结构"""
    is_homework: bool
# 守卫检查代理
guardrail_agent = Agent(
    name="Guardrail Check Agent",
    model="qwen-max",
    instructions="判断用户的问题是否属于家闲聊还是股票问题。如果是'chat'，应为 1，json 返回；如果是'auto_stock'，应为 2， json 返回",
    output_type=UserQuestionOutput,
)
# //闲聊agent
chat_agent = Agent(
    name="chat AI",
    model="qwen-max",
    handoff_description="负责聊天的大咖。",
    instructions="您是心理咨询师。请给用户以轻松愉悦积极健康的聊天内容和互动。",
)

# 股票agent
auto_stock_agent = Agent(
    name="autostock AI",
    model="qwen-max",
    handoff_description="负责选股。",
    instructions="您是专业选股操盘手，百发百中结合经济政治环境，突发事件，清晰地给用户选股建议。",
)

async def UserQuestion_guardrail(ctx, agent, input_data):
    """
    运行检查代理来判断输入是否为功课。
    如果不是功课 ('is_homework' 为 False)，则触发阻断 (tripwire)。
    """
    print(f"\n[Guardrail Check] 正在检查输入: '{input_data}'...")

    # 运行检查代理
    result = await Runner.run(guardrail_agent, input_data, context=ctx.context)

    # 解析输出
    final_output = result.final_output_as(UserQuestionOutput)

    auto_chat = not final_output.chat
    auto_stock = not final_output.auto_stock
    return GuardrailFunctionOutput(
        output_info=final_output,
        chat=auto_chat,
        auto_stock=auto_stock,
    )
async def setup():
    # await mcp.import_server(chat, prefix="")

    triage_agent = Agent(
        name="智能 Agent",
        model="qwen-max",
        instructions="您的任务是根据用户的提问，判断应该将请求分派给 '闲聊' 还是 '股票'。",
        handoffs=[chat_agent, auto_stock_agent],
        input_guardrails=[
            InputGuardrail(guardrail_function=UserQuestion_guardrail),
        ],
    )
async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("Available tools:", [t.name for t in tools])
        print("Available tools:", [t for t in tools])

if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)
