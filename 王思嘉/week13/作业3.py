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
        agent_name="小股助手",
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


async def chat(user_name:str, session_id: Optional[str], task: Optional[str], content: str, tools: List[str] = []):
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

    # 聊天工具
    tools1 = ['get_today_daily_news','get_douyin_hot_news','get_github_hot_news','get_toutiao_hot_news',
              'get_sports_news','get_today_familous_saying','get_today_motivation_saying',
              'get_today_working_saying','get_city_weather','get_address_detail','get_tel_info',
              'get_scenic_info','get_flower_info','get_rate_transform']
    # 股票工具
    tools2 = ['get_board_info','get_stock_rank','get_month_line','get_week_line','get_day_line','get_stock_info',
              'get_stock_minute_data']
    
    # mcp tools 选择
    if not tools or len(tools) == 0:
        tool_mcp_tools_filter: Optional[ToolFilterStatic] = None
    else:
    #     tool_mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools)
    # mcp_server = MCPServerSse(
    #     name="SSE Python Server",
    #     params={"url": "http://localhost:8900/sse"},
    #     cache_tools_list=False,
    #     tool_filter=tool_mcp_tools_filter,
    #     client_session_timeout_seconds=20,
    # )


    # 这里改成创建两个工具过滤器实例，用于控制不同类别工具的访问权限：
        tools1 = list(set(tools)&set(tools1))
        tools2 = list(set(tools)&set(tools2))
        tool_mcp_tools_filter1: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools1)
        tool_mcp_tools_filter2: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=tools2)

    # 闲聊mcp
    mcp_server1 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter1,
        client_session_timeout_seconds=20,
    )

    # 股票mcp
    mcp_server2 = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://localhost:8900/sse"},
        cache_tools_list=False,
        tool_filter=tool_mcp_tools_filter2,
        client_session_timeout_seconds=20,
    )


    # openai-agent支持的session存储，存储对话的历史状态
    session = AdvancedSQLiteSession(
        session_id=session_id, # 与 系统中的对话id 关联，存储在关系型数据库中
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

        result = Runner.run_streamed(agent, input=content, session=session) # 流式调用大模型

        assistant_message = ""
        async for event in result.stream_events():
            if event.type == "raw_response_event":
                if isinstance(event.data, ResponseTextDeltaEvent): # 如果式大模型的回答
                    if event.data.delta:
                        yield f"{event.data.delta}" # sse 不断发给前端
                        assistant_message += event.data.delta

        # 这一条大模型回答，存储对话
        append_message2db(session_id, "assistant", assistant_message)

    # 需要调用mcp 服务进行回答
    else:
        async with mcp_server1, mcp_server2:
            # 哪些工具直接展示结果
            need_viz_tools = ["get_month_line", "get_week_line", "get_day_line", "get_stock_minute_data"]
            if set(need_viz_tools) & set(tools):
                tool_use_behavior = "stop_on_first_tool" # 调用了tool，得到结果，就展示结果
            else:
                tool_use_behavior = "run_llm_again" # 调用了tool，得到结果，继续用大模型的总结结果

            chat_agent = Agent(
                name="ChatAssistant",
                instructions="角色：处理非股票相关问题的助手\n详细指令:\n",
                mcp_servers=[mcp_server1],
                model=OpenAIChatCompletionsModel(
                    model=os.environ["OPENAI_MODEL"],
                    openai_client=external_client,
                ),
                tool_use_behavior=tool_use_behavior,
                model_settings=ModelSettings(parallel_tool_calls=False)
            )

            stock_agent = Agent(
                name="StockAssistant",
                instructions="角色：处理非股票相关问题的助手\n详细指令:\n",
                mcp_servers=[mcp_server2],
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
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseOutputItemDoneEvent):
                    if isinstance(event.data.item, ResponseFunctionToolCall):
                        current_tool_name = event.data.item.name

                        # 工具名字、工具参数
                        yield "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"
                        assistant_message += "\n```json\n" + event.data.item.name + ":" + event.data.item.arguments + "\n" + "```\n\n"

                # run llm again 的回答： 基础tool的结果继续回答
                if event.type == "raw_response_event" and hasattr(event, 'data') and isinstance(event.data, ResponseTextDeltaEvent):
                    yield event.data.delta
                    assistant_message += event.data.delta


            append_message2db(session_id, "assistant", assistant_message)


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
                return [ChatSession(user_id = x.user_id, session_id=x.session_id, title=x.title, start_time=x.start_time) for x in chat_records]
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
