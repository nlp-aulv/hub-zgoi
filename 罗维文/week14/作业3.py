import asyncio
from fastmcp import Client
from sentence_transformers import SentenceTransformer
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.mcp import MCPServerSse, create_static_tool_filter

async def get_tools():
    tools_name = []
    tools_description = []
    client = Client("http://localhost:8900/sse")
    # 获取工具的名字和描述
    async with client:
        tools = await client.list_tools()  # 列举mcp server 中所有的tool
        for tool in tools:
            tools_name.append(tool.name)
            tools_description.append(tool.description)

    return tools_name, tools_description


async def rag_tools(query, tools_name, tools_description):
    if not tools_name:
        return []
    # 计算前三匹配的工具
    bge_model = SentenceTransformer('E:/AI学习/models/BAAI/bge-small-zh-v1.5/')

    query_embeddings = bge_model.encode(query, normalize_embeddings=True)
    tools_description_embeddings = bge_model.encode(tools_description, normalize_embeddings=True)

    score = query_embeddings @ tools_description_embeddings.T
    max_score = score.argsort()[::-1][:3]

    return [tools_name[i] for i in max_score]

async def main():
    query = "在给定温度22℃、降雨量120mm、施肥量50kg/亩的农业条件下，模型预测的单位面积作物产量是多少？"

    tools_name, tools_description = await get_tools()
    tools_name_rag = await rag_tools(query, tools_name, tools_description)

    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=create_static_tool_filter(allowed_tool_names=tools_name_rag),
        client_session_timeout_seconds=20,
    )

    external_client = AsyncOpenAI(
        api_key="",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    async with mcp_server:
        if not tools_name_rag:
            agent = Agent(
                name="Assistant",
                instructions="",
                # mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model="qwen-flash",
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model="qwen-flash",
                    openai_client=external_client,
                ),
                model_settings=ModelSettings(parallel_tool_calls=False)
            )
        result = Runner.run_streamed(agent, input=query)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
