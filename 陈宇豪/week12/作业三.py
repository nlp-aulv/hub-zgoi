import streamlit as st
import json

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

# 定义可用的MCP服务类别
MCP_CATEGORIES = {
    "news": "新闻服务",
    "saying": "名言服务", 
    "tool": "工具服务",
    "sentiment": "情感分析服务"
}

# 类别到服务器名称的映射
CATEGORY_TO_SERVER = {
    "news": "news_only_mcp",
    "saying": "saying_only_mcp",
    "tool": "tool_only_mcp",
    "sentiment": "sentiment_only_mcp"
}

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
    
    # 添加MCP服务选择功能
    st.subheader("MCP服务选择")
    st.caption("选择要启用的MCP服务")
    selected_categories = []
    for category, label in MCP_CATEGORIES.items():
        if st.checkbox(label, value=True, key=f"mcp_{category}"):
            selected_categories.append(category)
    
    # 将选择的服务存储在session_state中
    st.session_state['selected_mcp_categories'] = selected_categories
    
    if selected_categories:
        st.info(f"已选择服务: {', '.join([MCP_CATEGORIES[c] for c in selected_categories])}")
        
        # 如果只选择了一个类别，提示将使用专门的MCP服务器
        if len(selected_categories) == 1:
            category = selected_categories[0]
            st.info(f"将使用专门的 '{MCP_CATEGORIES[category]}' 服务器")
    else:
        st.warning("未选择任何服务")

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


async def get_model_response(prompt, model_name, use_tool):
    # 根据用户选择的类别来决定使用哪些MCP服务
    selected_categories = st.session_state.get('selected_mcp_categories', [])
    
    if use_tool and selected_categories:
        # 创建MCP服务器连接
        mcp_params = {
            "url": "http://localhost:8900/sse",
        }
        
        # 如果只选择了一个类别，则使用专门的MCP服务器
        if len(selected_categories) == 1:
            category = selected_categories[0]
            if category in CATEGORY_TO_SERVER:
                mcp_params["server_name"] = CATEGORY_TO_SERVER[category]
        # 如果选择了所有类别，则不指定特定服务器（使用默认的主服务器）
        elif selected_categories == list(MCP_CATEGORIES.keys()):
            pass  # 使用默认主服务器
        # 如果选择了部分类别（多于一个但不是全部），显示提示信息
        elif len(selected_categories) > 1:
            st.info(f"注意：当前选择了 {len(selected_categories)} 个服务类别，将使用完整工具集")
        
        async with MCPServerSse(
                name="SSE Python Server",
                params=mcp_params,
                client_session_timeout_seconds=20
        ) as mcp_server:
            # 显示将使用的工具类别
            categories_text = ', '.join([MCP_CATEGORIES[c] for c in selected_categories])
            st.info(f"将使用以下类别的工具: {categories_text}")
            
            external_client = AsyncOpenAI(
                api_key=key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            agent = Agent(
                name="Assistant",
                instructions="",
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=model_name,
                    openai_client=external_client,
                )
            )

            result = Runner.run_streamed(agent, input=prompt, session=session)
            async for event in result.stream_events():
                if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
    else:
        # 不使用工具或未选择任何类别
        external_client = AsyncOpenAI(
            api_key=key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
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