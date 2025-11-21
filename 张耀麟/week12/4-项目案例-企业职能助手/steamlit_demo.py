import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import ToolFilterContext
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
session = SQLiteSession("conversation_123")


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

    global session
    session = SQLiteSession("conversation_123")


def tool_filter(context: ToolFilterContext, tool) -> bool:
    '''
    for filter the tools for a specific agent
    '''
    if context.agent.name == "News Assistant":
        return "news" in tool.name

    if context.agent.name == "Normal Assistant":
        return "tools" in tool.name
    return False


async def get_model_response(prompt, model_name, use_tool):
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20,
            tool_filter=tool_filter
    ) as mcp_server:
        if use_tool:
            news_agent = Agent(
                name="News Assistant",
                instructions="You are an agent responsible for news",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
            )

            tools_agent = Agent(
                name="Normal Assistant",
                instructions="You are an agent which not for news",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
            )

            agent = Agent(
                name="Assistant",
                instructions="You determine which agent to use based on the user's question",
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                handoffs=[news_agent, tools_agent],
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


with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API Token已经配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token:', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

st.sidebar.button('清空聊天', on_click=clear_chat_history)

external_client = AsyncOpenAI(
    api_key=key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

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
