import streamlit as st
from agents.mcp.server import MCPServerSse
import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, SQLiteSession
from openai.types.responses import ResponseTextDeltaEvent
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api('chat_completions')
set_tracing_disabled(True)

st.set_page_config(page_title='企业职能机器人')
session = SQLiteSession('conversation_123')

with st.sidebar:
    st.title('职能AI+智能问答')
    if 'API_TOKEN' in st.session_state and len(st.session_state['API_TOKEN']) > 1:
        st.success('API TOKEN已配置', icon='✅')
        key = st.session_state['API_TOKEN']
    else:
        key = ""

    key = st.text_input('输入Token：', type='password', value=key)

    st.session_state['API_TOKEN'] = key
    model_name = st.selectbox('模型选择', ['qwen-flash', 'qwen-max'])

    st.info('系统将根据对话内容自动选择合适的工具！')

if 'messages' not in st.session_state.keys():
    st.session_state.messages = [
        {
            'role': 'assistant',
            'content': '你好，我是企业职能助手，可以AI对话，也可以根据你的需求自动调用对应的工具。'
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.write(message['content'])


def clear_chat_history():
    st.session_state.messages = [
        {
            'role': 'assistant',
            'content': '你好，我是企业职能助手，可以AI对话，也可以根据你的需求自动调用对应的工具。'
        }
    ]

    global session
    session = SQLiteSession('conversation_123')


st.sidebar.button('清空聊天记录', on_click=clear_chat_history)

async def get_model_response(prompt, model_name):
    async with MCPServerSse(
        name='SSE Python Server',
        params={
            'url': 'http://localhost:8900/sse',
        },
        client_session_timeout_seconds=20
    ) as mcp_server:
        external_client = AsyncOpenAI(
            api_key=key,
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
        )

        agent = Agent(
            name='Assistant',
            instructions="""你是一个职能助手，能够根据用户的问题意图自动选择合适的工具。
            请分析用户问题，判断是否需要使用工具以及使用哪个工具：
            - 如果用户询问 每日新闻简报、抖音热搜榜、GitHub热榜、头条新闻、电竞资讯，使用news工具
            - 如果用户询问 经典语句、励志古言、心灵鸡汤，使用saying工具
            - 如果用户 询问天气、物流地址解析、手机归属地、旅游景区、花语箴言、汇率，使用tool工具
            - 如果用户进行 文本情感分析，使用sentiment工具
            - 如果用户的问题不是以上情况，则直接回答
            
            请根据对话内容自动判断并选择合适的工具。""",
            mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=external_client,
            )
        )

        result = Runner.run_streamed(agent, input=prompt, session=session)
        async for event in result.stream_events():
            if event.type == 'raw_response_event' and isinstance(event.data, ResponseTextDeltaEvent):
                yield event.data.delta


if len(key) > 1:
    if prompt := st.chat_input():
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        with st.chat_message('user'):
            st.markdown(prompt)

        with st.chat_message('assistant'):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner('请求中...'):
                try:
                    response_genetator = get_model_response(prompt, model_name)

                    async def stream_and_accumulate(generator):
                        accumulated_text = ""
                        async for chunk in generator:
                            accumulated_text += chunk
                            message_placeholder.markdown(accumulated_text + '▌▌')
                        return accumulated_text

                    full_response = asyncio.run(stream_and_accumulate(response_genetator))
                    message_placeholder.markdown(full_response)

                except Exception as e:
                    error_message = f"发生错误：{e}"
                    message_placeholder.error(error_message)
                    full_response = error_message
                    print('Error during streaming: ', e)

            st.session_state.messages.append({'role': 'assistant', 'content': full_response})
