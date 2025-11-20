import streamlit as st
from agents.mcp import ToolFilterContext
from agents.mcp.server import MCPServerSse,MCPTool
import asyncio

from jieba.lac_small.predict import results
from openai.types.responses import ResponseTextDeltaEvent
from openai.types.responses import ResponseTextDeltaEvent
from agents import (
    set_default_openai_api,
    set_tracing_disabled,
    AsyncOpenAI,
    Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel,
)
from dotenv import load_dotenv
from fastmcp import Client, FastMCP
import os
from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp

set_tracing_disabled(True)

load_dotenv()

api_url = os.getenv("API_URL")
session = SQLiteSession("conversation_123")


with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox("选择模型", ["deepseek-chat"])
    use_tool = st.checkbox("使用工具")

    if use_tool:
        st.write("MCP 服务列表")
        news_server = st.checkbox("新闻服务")
        tool_server = st.checkbox("工具服务")
        saying_server = st.checkbox("名言服务")

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话也可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话也可以调用内部工具。"}]

    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)

async def tool_filter(context:ToolFilterContext, tool:MCPTool) -> bool:
    found = False # 服务中是否包含工具
    # 勾选了并且结果是未找到
    if news_server and not found:
        found = tool_in_mcp(news_mcp, tool.name)

    if tool_server and not found:
        found = tool_in_mcp(tool_mcp, tool.name)

    if saying_server and not found:
        found = tool_in_mcp(saying_mcp, tool.name)

    return found

async def tool_in_mcp(mcp:FastMCP, tool_name:str) -> bool:
    async with Client(mcp) as client:
        available_tools = await client.list_tools()
        available_tool_names = [t.name for t in available_tools]
        if tool_name in available_tool_names: return True
        else: return False


async def get_model_response(prompt, model_name, use_tool):
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            tool_filter=tool_filter,
            client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url=api_url,
        )
        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="你是一个企业职能助手，能够根据用户的提问和提供的工具回答问题。如果调用了工具，请列出调用的工具列表。",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

        result = Runner.run_streamed(agent, input=prompt, session=session)

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta

if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner("请求中..."):
                try:
                    response_generator = get_model_response(prompt, model_name, use_tool)

                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + "▌")
                        return accumulated_text

                    full_response = asyncio.run(stream_and_accumulate(response_generator))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误: {e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print(f"Error during streaming: {e}")

            # 4. 将完整的助手回复添加到 session state
            st.session_state.messages.append({"role": "assistant", "content": full_response})