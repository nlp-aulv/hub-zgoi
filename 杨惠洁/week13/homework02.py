import os
import random
import string
from datetime import datetime
from typing import List, Dict, Any, Optional

from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from typing import AsyncGenerator

from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI
from openai.types.responses import ResponseTextDeltaEvent, ResponseOutputItemDoneEvent, ResponseFunctionToolCall
from jinja2 import Environment, FileSystemLoader

from models.data_models import ChatSession
from models.orm import ChatSessionTable, ChatMessageTable, SessionLocal, UserTable
from fastapi.responses import StreamingResponse


def generate_random_chat_id(length=12):
    with SessionLocal() as session:
        for retry_time in range(20):
            characters = string.ascii_letters + string.digits
            session_id = ''.join(random.choice(characters) for i in range(length))
            chat_session_record: ChatSessionTable | None = session.query(ChatSessionTable).filter(
                ChatSessionTable.session_id == session_id).first()
            if chat_session_record is None:
                break

            if retry_time > 10:
                raise Exception("Failed to generate a unique session_hash")

    return session_id


def get_init_message(
        task: str,
) -> List[Dict[Any, Any]]:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("chat_start_system_prompt.jinjia2")

    if task == "股票分析":
        task_description = """
1. 专注于全球主要股票市场（如 NYSE, NASDAQ, SHSE, HKEX）的分析。
2. 必须使用专业、严谨的金融术语，如 P/E, EPS, Beta, ROI, 护城河 (Moat) 等。
3. **在提供分析时，必须清晰地说明数据来源、分析模型的局限性，并强调你的意见不构成最终的投资建议。**
4. 仅基于公开市场数据和合理的财务假设进行分析，禁止进行内幕交易或非公开信息的讨论。
5. 结果要求：提供结构化的分析（如：公司概览、财务健康度、估值模型、风险与机遇）。
"""
    elif task == "数据BI":
        task_description = """
1. 帮助用户理解他们的数据结构、商业指标和关键绩效指标 (KPI)。
2. 用户的请求通常是数据查询、指标定义或图表生成建议。
3. **关键约束：你的输出必须是可执行的代码块 (如 SQL 或 Python)，或者清晰的逻辑步骤，用于解决用户的数据问题。**
4. 严格遵守数据分析的逻辑严谨性，确保每一个结论都有数据支撑。
5. 当被要求提供可视化建议时，请推荐最合适的图表类型（如：时间序列用折线图，分类对比用柱状图）。"""
    else:
        task_description = """
1. 保持对话的自然和流畅，以轻松愉快的语气回应用户。
2. 避免过于专业或生硬的术语，除非用户明确要求。
3. 倾听用户的表达，并在适当的时候提供支持、鼓励或趣味性的知识。
4. 确保回答简洁，富有情感色彩，不要表现得像一个没有感情的机器。
5. 关键词：友好、轻松、富有同理心。
        """

    system_prompt = template.render(
        agent_name="小呆助手",
        task_description=task_description,
        current_datetime=datetime.now(),
    )
    return system_prompt


def init_chat_session(
        user_name: str,
        user_question: str,
        session_id: str,
        task: str,
) -> str:
    # 创建对话的title，通过summary agent
    # 存储数据库
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()

        chat_session_record = ChatSessionTable(
            user_id=user_id[0],
            session_id=session_id,
            title=user_question,
        )
        print("add ChatSessionTable", user_id[0], session_id)
        session.add(chat_session_record)
        session.commit()
        session.flush()

        message_recod = ChatMessageTable(
            chat_id=chat_session_record.id,
            role="system",
            content=get_init_message(task)
        )
        session.add(message_recod)
        session.flush()
        session.commit()

    return True


def create_chat_agent(external_client, instructions, mcp_servers=None, tool_use_behavior="stop_on_first_tool"):
    """创建通用的聊天Agent"""
    return Agent(
        name="ChatAssistant",
        instructions=instructions,
        mcp_servers=mcp_servers,
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        tool_use_behavior=tool_use_behavior,
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_stock_agent(external_client, mcp_servers):
    """创建专业的股票分析Agent"""
    stock_instructions = """
你是一个专业的股票分析师，专注于全球主要股票市场的分析。

核心能力：
1. 股票数据查询和分析
2. 技术指标解读
3. 财务数据分析
4. 市场趋势判断

专业要求：
- 必须使用专业金融术语（P/E, EPS, Beta, ROI, 护城河等）
- 必须说明数据来源和分析局限性
- 严禁提供投资建议，只做客观分析
- 基于公开市场数据和合理财务假设

当用户询问股票相关问题时，请使用专业的股票分析工具进行深入分析。
"""

    return Agent(
        name="StockAnalyst",
        instructions=stock_instructions,
        mcp_servers=mcp_servers,
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        tool_use_behavior="stop_on_first_tool",
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


def create_router_agent(external_client):
    """创建路由Agent，决定由哪个专业Agent处理问题"""
    router_instructions = """
你是一个智能路由助手，负责分析用户问题并将其分配给最合适的专业助手。

可用的专业助手：
1. 股票分析师 - 处理所有股票、投资、金融市场相关的问题
2. 聊天助手 - 处理日常对话、闲聊、一般性问题

路由规则：
- 如果问题涉及：股票、股价、K线、财报、投资、基金、证券、市场行情 → 转给股票分析师
- 如果问题涉及：日常聊天、生活咨询、学习建议、工作问题 → 转给聊天助手
- 如果无法确定，优先转给聊天助手

请只输出路由决策，不要回答问题本身。
输出格式：{"agent": "stock"} 或 {"agent": "chat"}
"""

    return Agent(
        name="Router",
        instructions=router_instructions,
        model=OpenAIChatCompletionsModel(
            model=os.environ["OPENAI_MODEL"],
            openai_client=external_client,
        ),
        model_settings=ModelSettings(parallel_tool_calls=False)
    )


async def chat_with_handoff(user_name: str, session_id: Optional[str], task: Optional[str], content: str,
                            tools: List[str] = []):
    """使用Handoff机制的协作对话"""

    # 初始化会话（如果不存在）
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 存储用户消息
    append_message2db(session_id, "user", content)

    # 初始化客户端和工具
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # MCP工具配置
    tool_mcp_tools_filter = None
    mcp_servers = []
    if tools and len(tools) > 0:
        tool_mcp_tools_filter = ToolFilterStatic(allowed_tool_names=tools)
        mcp_server = MCPServerSse(
            name="SSE Python Server",
            params={"url": "http://localhost:8900/sse"},
            cache_tools_list=False,
            tool_filter=tool_mcp_tools_filter,
            client_session_timeout_seconds=20,
        )
        mcp_servers = [mcp_server]

    # 会话存储
    session = AdvancedSQLiteSession(
        session_id=session_id,
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 如果没有指定工具，使用Agent协作系统
    if not tools or len(tools) == 0:
        async for chunk in run_agent_collaboration(external_client, content, session, session_id):
            yield chunk
    else:
        # 使用原有逻辑（带工具的单一Agent）
        async for chunk in run_single_agent_with_tools(external_client, content, session, session_id, mcp_servers, task,
                                                       tools):
            yield chunk


async def run_agent_collaboration(external_client, content, session, session_id):
    """运行Agent协作系统"""

    # 1. 创建路由Agent
    router_agent = create_router_agent(external_client)

    # 2. 路由决策
    router_result = await Runner.run(router_agent, input=content)
    router_response = router_result.final_output

    # 解析路由决策
    agent_choice = "chat"  # 默认使用聊天助手

    try:
        # 尝试从路由响应中提取决策
        if "stock" in router_response.lower() or "股票" in router_response or "投资" in router_response:
            agent_choice = "stock"
        elif "chat" in router_response.lower() or "聊天" in router_response:
            agent_choice = "chat"
    except:
        pass

    # 3. 根据路由选择执行相应的Agent
    if agent_choice == "stock":
        async for chunk in run_stock_agent(external_client, content, session, session_id):
            yield chunk
    else:
        async for chunk in run_chat_agent(external_client, content, session, session_id):
            yield chunk


async def run_stock_agent(external_client, content, session, session_id):
    """运行股票分析Agent"""

    # 创建股票专用的MCP工具
    stock_tools = ["get_day_line", "get_week_line", "get_month_line", "get_stock_minute_data"]
    tool_mcp_tools_filter = ToolFilterStatic(allowed_tool_names=stock_tools)

    mcp_server = MCPServerSse(
        name="StockTools",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    async with mcp_server:
        stock_agent = create_stock_agent(external_client, [mcp_server])
        result = Runner.run_streamed(stock_agent, input=content, session=session)

        assistant_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):
                    if event.data.delta:
                        yield f"{event.data.delta}"
                        assistant_message += event.data.delta

        append_message2db(session_id, "assistant", assistant_message)


async def run_chat_agent(external_client, content, session, session_id):
    """运行闲聊Agent"""

    chat_instructions = """
你是一个友好、热情的聊天助手，专注于日常对话和一般性问题。

对话风格：
- 保持自然、流畅、轻松愉快的语气
- 富有同理心，积极倾听用户表达
- 在适当时候提供支持、鼓励或趣味性知识
- 回答简洁明了，避免过于专业的术语

擅长领域：
- 日常聊天和情感支持
- 学习建议和工作咨询
- 生活技巧和兴趣爱好
- 一般性知识问答
"""

    chat_agent = create_chat_agent(external_client, chat_instructions)
    result = Runner.run_streamed(chat_agent, input=content, session=session)

    assistant_message = ""
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                if event.data.delta:
                    yield f"{event.data.delta}"
                    assistant_message += event.data.delta

    append_message2db(session_id, "assistant", assistant_message)


async def run_single_agent_with_tools(external_client, content, session, session_id, mcp_servers, task, tools):
    """原有的带工具的单Agent逻辑"""
    instructions = get_init_message(task)

    need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
    if set(need_viz_tools) & set(tools):
        tool_use_behavior = "stop_on_first_tool"
    else:
        tool_use_behavior = "run_llm_again"

    async with mcp_servers[0]:
        agent = create_chat_agent(external_client, instructions, mcp_servers, tool_use_behavior)
        result = Runner.run_streamed(agent, input=content, session=session)

        assistant_message = ""
        current_tool_name = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                            ResponseOutputItemDoneEvent):
                if isinstance(event.data.item, ResponseFunctionToolCall):
                    current_tool_name = event.data.item.name
                    yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                    assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

            if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                            ResponseTextDeltaEvent):
                yield event.data.delta
                assistant_message += event.data.delta

        append_message2db(session_id, "assistant", assistant_message)


async def run_legacy_chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str,
                          tools: List[str] = []):
    """原有的chat函数逻辑 - 保持原样"""
    # 对话管理，通过session id
    if session_id:
        with SessionLocal() as session:
            record = session.query(ChatSessionTable).filter(ChatSessionTable.session_id == session_id).first()
            if not record:
                init_chat_session(user_name, content, session_id, task)

    # 对话记录，存关系型数据库
    append_message2db(session_id, "user", content)

    # 获取system message，需要传给大模型，并不能给用户展示
    instructions = get_init_message(task)

    # agent 初始化
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    # mcp tools 选择
    if not tools or len(tools) == 0:
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
        tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter,
        client_session_timeout_seconds=20,
    )

    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id,  # 与 系统中的对话id 关联，存储在关系型数据库中
        db_path="./assert/conversations.db",
        create_tables=True
    )

    # 如果没有选择工具，默认直接调用大模型回答
    if not tools or len(tools) == 0:
        agent = Agent(
            name="Assistant",
            instructions=instructions,
            # mcp_servers=[mcp_server],
            model=OpenAIChatCompletionsModel(
                model=os.environ["OPENAI_MODEL"],
                openai_client=external_client,
            ),
            # tool_use_behavior="stop_on_first_tool",
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        result = Runner.run_streamed(agent, input=content, session=session)  # 流式调用大模型

        assistant_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent):  # 如果式大模型的回答
                    if event.data.delta:
                        yield f"{event.data.delta}"  # sse 不断发给前端
                        assistant_message += event.data.delta

        # 这一条大模型回答，存储对话
        append_message2db(session_id, "assistant", assistant_message)

    # 需要调用mcp 服务进行回答
    else:
        async with mcp_server:
            # 哪些工具直接展示结果
            need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
            if set(need_viz_tools) & set(tools):
                tool_use_behavior = "stop_on_first_tool"  # 调用了tool，得到结果，就展示结果
            else:
                tool_use_behavior = "run_llm_again"  # 调用了tool，得到结果，继续用大模型的总结结果

            agent = Agent(
                name="Assistant",
                instructions=instructions,
                mcp_servers=[mcp_server],
                model=OpenAIChatCompletionsModel(
                    model=os.environ["OPENAI_MODEL"],
                    openai_client=external_client,
                ),
                tool_use_behavior=tool_use_behavior,
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            result = Runner.run_streamed(agent, input=content, session=session)

            assistant_message = ""
            current_tool_name = ""
            async for event in result.stream_events():
                # if event.type == "run_item_stream_event" and hasattr(event, 'name') and event.name == "tool_output" and current_tool_name not in need_viz_tools:
                #     yield event.item.raw_item["output"]
                #     assistant_message += event.item.raw_item["output"]

                # tool_output
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                                ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name

                        # 工具名字、工具参数
                        yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                        assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                # run llm again 的回答： 基础tool的结果继续回答
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data,
                                                                                                ResponseTextDeltaEvent):
                    yield event.data.delta
                    assistant_message += event.data.delta

            append_message2db(session_id, "assistant", assistant_message)


async def chat(user_name: str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
    """增强版的chat函数，支持Agent协作"""

    # 如果明确指定了任务类型，使用原有逻辑
    if task and task in ["股票分析", "数据BI"]:
        async for chunk in run_legacy_chat(user_name, session_id, task, content, tools):
            yield chunk
    else:
        # 使用新的Agent协作系统
        async for chunk in chat_with_handoff(user_name, session_id, task, content, tools):
            yield chunk

def get_chat_sessions(session_id: str) -> List[Dict[str, Any]]:
    with SessionLocal() as session:

        chat_messages: Optional[List[ChatMessageTable]] = session.query(ChatMessageTable) \
            .join(ChatSessionTable) \
            .filter(
            ChatSessionTable.session_id == session_id
        ).all()

        result = []
        if chat_messages:
            for record in chat_messages:
                result.append({
                    "id": record.id, "create_time": record.create_time,
                    "feedback": record.feedback, "feedback_time": record.feedback_time,
                    "role": record.role, "content": record.content
                })

        return result


def delete_chat_session(session_id: str) -> bool:
    with SessionLocal() as session:
        session_id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if session_id is None:
            return False

        session.query(ChatMessageTable).where(ChatMessageTable.chat_id == session_id[0]).delete()
        session.query(ChatSessionTable).where(ChatSessionTable.id == session_id[0]).delete()
        session.commit()

    return True


def change_message_feedback(session_id: str, message_id: int, feedback: bool) -> bool:
    with SessionLocal() as session:
        id = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if id is None:
            return False

        record = session.query(ChatMessageTable).filter(ChatMessageTable.id == message_id,
                                                        ChatMessageTable.chat_id == id[0]).first()
        if record is not None:
            record.feedback = feedback
            record.feedback_time = datetime.now()
            session.commit()

        return True


def list_chat(user_name: str) -> Optional[List[Any]]:
    with SessionLocal() as session:
        user_id = session.query(UserTable.id).filter(UserTable.user_name == user_name).first()
        if user_id:
            chat_records: Optional[List[ChatSessionTable]] = session.query(
                ChatSessionTable.user_id,
                ChatSessionTable.session_id,
                ChatSessionTable.title,
                ChatSessionTable.start_time).filter(ChatSessionTable.user_id == user_id[0]).all()
            if chat_records:
                return [ChatSession(user_id=x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time)
                        for x in chat_records]
            else:
                return []
        else:
            return []


def append_message2db(session_id: str, role: str, content: str) -> bool:
    with SessionLocal() as session:
        message_recod = session.query(ChatSessionTable.id).filter(ChatSessionTable.session_id == session_id).first()
        if message_recod:
            message_recod = ChatMessageTable(
                chat_id=message_recod[0],
                role=role,
                content=content
            )
            session.add(message_recod)
            session.commit()
