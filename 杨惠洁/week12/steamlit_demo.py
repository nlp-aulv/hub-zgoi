import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents.mcp import MCPServer
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

st.set_page_config(page_title="企业职能机器人")
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

    model_name = st.selectbox("选择模型", ["qwen-flash", "qwen-max"])
    use_tool = st.checkbox("使用工具")

    # 添加工具类型选择
    if use_tool:
        tool_type = st.radio(
            "选择工具类型:",
            ["全部工具", "新闻工具","情绪工具", "其他工具"],
            help="新闻工具: 只调用新闻相关功能\n其他工具: 排除新闻功能"
        )
    else:
        tool_type = "无工具"

# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}]
    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)


def create_tool_filter(tool_type):
    """根据选择的工具类型创建过滤器"""
    if tool_type == "新闻工具":
        # 只允许包含"新闻"关键词的工具
        def news_filter(tool_name, tool_description):
            return "新闻" in tool_name or "新闻" in tool_description or "news" in tool_name.lower()

        return news_filter

    elif tool_type == "情绪工具":
        # 只允许包含"情绪"关键词的工具
        def emotion_filter(tool_name, tool_description):
            return "情绪" in tool_name or "情绪" in tool_description or "emotion" in tool_name.lower()

        return emotion_filter

    elif tool_type == "其他工具":
        # 排除新闻相关的工具
        def other_tools_filter(tool_name, tool_description):
            return not ("新闻" in tool_name or "新闻" in tool_description or "news" in tool_name.lower())

        return other_tools_filter

    elif tool_type == "全部工具":
        # 允许所有工具
        def all_tools_filter(tool_name, tool_description):
            return True

        return all_tools_filter

    else:
        # 无工具
        return None


async def get_model_response(prompt, model_name, use_tool, tool_type):
    # 创建工具过滤器
    tool_filter = create_tool_filter(tool_type) if use_tool else None

    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions=f"""
                你是一个企业职能助手，根据用户需求选择合适的工具。
                当前可用的工具类型: {tool_type}

                注意：请根据工具类型限制来选择合适的工具：
                - 如果用户询问新闻相关内容，请使用新闻工具
                - 如果用户询问情绪类内容，请使用情绪工具
                - 如果用户询问其他功能，请使用相应的工具
                - 确保不超出当前允许的工具范围
                """,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                ),
                tool_filter=tool_filter  # 添加工具过滤器
            )
        else:
            agent = Agent(
                name="Assistant",
                instructions="你是一个企业职能助手，请用对话方式回答用户问题。",
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
                    response_generator = get_model_response(prompt, model_name, use_tool, tool_type)


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

            st.session_state.messages.append({"role": "assistant", "content": full_response})