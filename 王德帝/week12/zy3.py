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
session = SQLiteSession("conversation_123")  # openai agent 提供的 基于内存的上下文缓存

# streamlit
# session_state 当前对话的缓存
# session_state.messages 此次对话的历史上下文

# 页面的侧边栏
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

    # 显示工具列表
    tool_list_container = st.empty()


# 初始化的对话
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [
        {"role": "system", "content": "你好，我是企业职能助手，可以AI对话 也 可以调用内部工具。"}
    ]

    global session
    session = SQLiteSession("conversation_123")


st.sidebar.button('清空聊天', on_click=clear_chat_history)

#获取当前可用的工具列表
async def load_tools():
    """
    从 MCP Server 获取当前可用的工具列表
    """
    async with MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
    ) as mcp_server:

        tools = await mcp_server.list_tools()
        #Available tools: ['get_today_daily_news', 'get_douyin_hot_news', 'get_github_hot_news', 'get_toutiao_hot_news', 'get_sports_news', 'get_today_familous_saying', 'get_today_motivation_saying', 'get_today_working_saying', 'get_city_weather', 'get_address_detail', 'get_tel_info', 'get_scenic_info', 'get_flower_info', 'get_rate_transform', 'sentiment_classification', 'summary_work_history']
        return tools

def filter_news_tools(tools):
    """筛选新闻类工具"""
    news_tools = ['get_today_daily_news', 'get_douyin_hot_news', 'get_github_hot_news', 
                  'get_toutiao_hot_news', 'get_sports_news']
    return [tool for tool in tools if tool.name in news_tools]

def filter_utility_tools(tools):
    """筛选实用工具类"""
    utility_tools = ['get_city_weather', 'get_address_detail', 'get_tel_info', 
                     'get_scenic_info', 'get_flower_info', 'get_rate_transform', 
                     'sentiment_classification', 'summary_work_history']
    return [tool for tool in tools if tool.name in utility_tools]

async def get_model_response(prompt, model_name, use_tool):
    """
    prompt 当前用户输入
    model_name 模型版本
    use_tool 是否调用工具
    """
    async with MCPServerSse(
            name="SSE Python Server",
            params={
                "url": "http://localhost:8900/sse",
            },
            cache_tools_list=False, # 如果 True 第一次调用后，缓存mcp server 所有工具信息，不再进行list tool
            # tool_filter 对tool筛选（可以写一个函数筛选，也可以通过黑名单/白名单筛选）
            # client_session_timeout_seconds 超时时间
            client_session_timeout_seconds=20,
            tool_filter=lambda tools: filter_news_tools(tools) if "新闻" in prompt else filter_utility_tools(tools)
    ) as mcp_server:
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

        # session openai-agent 中 缓存的上下文
        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


if len(key) > 1:
    # ===== 新增：侧边栏加载工具列表 =====
    if use_tool:
        try:
            tools = asyncio.run(load_tools())
            tool_list_container.markdown("### 🔧 可用工具列表")
            tools_info = []  # 收集所有工具信息
            for tool in tools:
                tools_info.append(f"""**{tool.name}**""")
            tool_list_container.markdown("\n".join(tools_info))
        except Exception as e:
            tool_list_container.error(f"加载工具失败：{e}")

    else:
        tool_list_container.markdown("")  # 不勾选时清空显示

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"): # 用户输入
            st.markdown(prompt)

        with st.chat_message("assistant"): # 大模型输出
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
