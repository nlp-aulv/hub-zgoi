import os
os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import asyncio
from fastmcp import Client
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
import numpy as np

from agents.mcp.server import MCPServerSse
from openai.types.responses import ResponseContentPartDoneEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig, ModelSettings, RawResponsesStreamEvent
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import ToolFilterStatic
from agents import set_default_openai_api, set_tracing_disabled
set_default_openai_api("chat_completions")
set_tracing_disabled(True)

client = Client("http://localhost:8900/sse")

async def list_tools():
    async with client:
        result = await client.list_tools()
        names = [tool.name for tool in result]
        descs = [tool.description for tool in result]
        return names, descs

async def rag_tool_call():
    function_names, function_descriptions = await list_tools()
    print("ALL tools")
    print(function_names)

    tfidf = TfidfVectorizer()
    function_descriptions_tfidf = tfidf.fit_transform(
        [" ".join(jieba.lcut(x)) for x in function_descriptions]
    ).toarray()

    user_question = "计算农产品在零售价格100，生产成本80，日销量100下的利润。"
    user_question_tfidf = tfidf.transform(
        [" ".join(jieba.lcut(user_question))]
    ).toarray()

    ids = np.dot(user_question_tfidf, function_descriptions_tfidf.T)[0]
    ids_order = np.argsort(ids)[::-1][:5]

    print("Top 5 candidate tool")
    for i in ids_order:
        print(function_names[i])

    tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=[function_names[i] for i in ids_order[:5]])
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    async with mcp_server:
        agent = Agent(
            name="Tool agent",
            instructions="调用工具解决问题，首先复述用户问题，然后说明调用的工具，然后直接输出工具输出结果，最终总结。",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model="qwen-max",
                openai_client=external_client,
            ),
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        result = Runner.run_streamed(agent, input=user_question, run_config=RunConfig(model_settings=ModelSettings(parallel_tool_calls=False)))
        async for event in result.stream_events():

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    print("工具信息", event.data.item, end="\n", flush=True)

            if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output":
                print("Agent 工具调用", event, end="\n", flush=True)

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(rag_tool_call())