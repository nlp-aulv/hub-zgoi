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
    # use_tool = st.checkbox("使用工具")
    # if use_tool:
    news_tool = st.checkbox("查询新闻")
    tool_tool = st.checkbox("调用工具")
    saying_tool = st.checkbox("生成句子")
    use_tool = False
    if news_tool or tool_tool or saying_tool:
        use_tool = True


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

async def get_model_response(prompt, model_name, use_tool,news_tool,tool_tool,saying_tool):
    # filter_tools=[]
    # if news_tool:
    #     filter_tools = ["get_today_daily_news", "get_douyin_hot_news", "get_github_hot_news", "get_toutiao_hot_news", "get_sports_news"]
    # if tool_tool:
    #     filter_tools += ["get_city_weather", "get_address_detail", "get_tel_info", "get_scenic_info", "get_flower_info", "get_rate_transform", "sentiment_classification"]
    # if saying_tool:
    #     filter_tools += ["get_today_familous_saying", "get_today_motivation_saying", "get_today_working_saying"]
    def my_filter(context, tool) -> bool:
        # 检查tool对象是否有name属性，并且针对工具名称进行过滤
        tool_name = getattr(tool, 'name', '') if hasattr(tool, 'name') else str(tool)
        
        if news_tool and "news" in tool_name:
            return True
        elif saying_tool and "saying" in tool_name:
            return True
        elif tool_tool and not ("news" in tool_name or "saying" in tool_name):
            # 当选择tool_tool时，只允许既不是新闻也不是句子生成的工具
            return True
        return False
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            tool_filter=my_filter,
            client_session_timeout_seconds=20
    )as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if use_tool:
            agent = Agent(
                name="Assistant",
                instructions="",
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
                    response_generator = get_model_response(prompt, model_name, use_tool,news_tool,tool_tool,saying_tool)

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
