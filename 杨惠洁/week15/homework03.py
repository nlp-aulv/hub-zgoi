import asyncio
import os

os.environ["OPENAI_API_KEY"] = "sk-c4395731abd4446b8642c7734c8dbf56"
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

import json
import requests
import urllib.parse
from typing import List, Dict, Any, Tuple

from agents import Agent, function_tool, AsyncOpenAI, OpenAIChatCompletionsModel, ModelSettings, Runner, \
    set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)

MODEL_NAME = "qwen-max"
API_KEY = os.getenv("OPENAI_API_KEY", "sk-c4395731abd4446b8642c7734c8dbf56")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

# 初始化 AsyncOpenAI 客户端
llm_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL
)

# 定义模型设置
model_settings = ModelSettings(
    model=MODEL_NAME,
    client=llm_client,
    temperature=0.3
)

JINA_API_KEY = "jina_8918effb420d4bff8530c9d9f3bbe536NWhiCZdKQFNgoFLd4aganV1XnsaA"


def search_jina(query: str) -> str:
    """通过jina进行谷歌搜索，返回JSON格式的搜索结果字符串"""
    print(f"-> [Jina Search] 正在搜索: {query[:50]}...")
    try:
        encoded_query = urllib.parse.quote(query)
        url = f"https://s.jina.ai/?q={encoded_query}&hl=zh-cn"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "no-content"
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        results = response.json().get('data', [])
        formatted_results = [
            {"title": res.get("title", ""), "url": res.get("url", ""), "snippet": res.get("content", "")} for res in
            results]
        return json.dumps(formatted_results, ensure_ascii=False)
    except requests.exceptions.RequestException as e:
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e), "query": query}, ensure_ascii=False)


def crawl_jina(url: str) -> str:
    """通过jina抓取完整网页内容，返回Markdown格式的文本"""
    print(f"-> [Jina Crawl] 正在抓取: {url[:50]}...")
    try:
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {JINA_API_KEY}",
            "X-Respond-With": "content",
            "X-Content-Type": "markdown"
        }
        response = requests.get("https://r.jina.ai/" + url, headers=headers, timeout=20)
        response.raise_for_status()
        content = response.json().get("data", {}).get("content", f"无法抓取 URL: {url} 的内容。")
        return content
    except requests.exceptions.RequestException as e:
        return f"抓取失败: {e}"
    except Exception as e:
        return f"抓取失败: {e}"


async def async_search_jina(query: str) -> str:
    return await asyncio.to_thread(search_jina, query)


async def async_crawl_jina(url: str) -> str:
    return await asyncio.to_thread(crawl_jina, url)


external_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# --- 3. 代理定义 (新增 Reviewer Agent) ---

orchestrator_system_prompt = """
... (保持不变，DeepResearchAgent 职责：规划和整合) ...
"""
DeepResearchAgent = Agent(
    "Deep Research Orchestrator",
    instructions=orchestrator_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

drafting_system_prompt = """
... (保持不变，DraftingAgent 职责：撰写内容) ...
"""
DraftingAgent = Agent(
    "Content Drafting Specialist",
    instructions=drafting_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)

# **新增 Reviewer Agent**
reviewer_system_prompt = """
你是一名严谨的报告审阅专家，你的任务是评估提供的报告章节草稿。
你必须遵循 ReAct 机制，在分析内容后，给出下一步的行动指导。

**你的输出必须包含以下三个部分：**
1. **Thought (思考):** 评估报告的质量、准确性、逻辑连贯性和对主题的覆盖程度。
2. **Action (行动):** 决定下一步的行动：'FINISH'（如果草稿足够好） 或 'REVISE'（如果需要修改）。
3. **Observation (观察/建议):**
    - 如果 Action 是 'FINISH'，输出简短的确认信息。
    - 如果 Action 是 'REVISE'，输出具体的、结构化的改进建议和修改要求。

**输出示例 (Action: REVISE):**
Thought: 初稿内容有些过于笼统，缺乏具体的数据支持，且对核心概念的解释不够深入。
Action: REVISE
Observation: 请在以下方面改进： 1. 增加至少一个具体的应用案例。 2. 引入文中提到的核心技术 A 的数据或市场份额。 3. 确保引用格式统一。
"""
ReviewerAgent = Agent(
    "Report Reviewer",
    instructions=reviewer_system_prompt,
    model=OpenAIChatCompletionsModel(
        model="qwen-max",
        openai_client=external_client,
    ),
)


# --- 5. 新增章节处理函数（用于并行） ---

async def process_section_with_react(
        section_data: Dict[str, str],
        i: int,
        max_revisions: int = 2
) -> Tuple[str, str]:
    """
    处理单个章节，包括检索、抓取、初始起草，并进行基于 Reviewer Agent 的 ReAct 迭代。
    返回: (section_title, final_draft_markdown)
    """
    section_title = section_data.get("section_title")
    search_keywords = section_data.get("search_keywords")
    print(f"\n--- Step 3.{i + 1}: 正在启动章节并行处理: {section_title} ---")

    # 3.1. 精确检索
    section_query = f"{section_title} 搜索关键词: {search_keywords}"
    section_search_results_str = await async_search_jina(section_query)

    # 3.2. 筛选并抓取前2个链接
    try:
        search_results = json.loads(section_search_results_str)
        urls_to_crawl = [res['url'] for res in search_results if res.get('url')][:2]
    except:
        urls_to_crawl = []

    crawled_content = []
    for url in urls_to_crawl:
        content = await async_crawl_jina(url)
        crawled_content.append(f"--- URL: {url} ---\n{content[:3000]}...\n")

    raw_materials = "\n\n".join(crawled_content)

    # 3.3. 迭代起草 (ReAct 循环)
    current_draft = ""
    revision_count = 0

    while revision_count < max_revisions + 1:

        print(f"-> {section_title}: 正在进行第 {revision_count + 1} 次起草/修订...")

        # 第一次是起草，之后是修订
        if revision_count == 0:
            draft_prompt = f"""
            **章节主题:** {section_title}
            **搜索结果摘要:** {section_search_results_str[:3000]}...
            **原始网页内容 (请基于此内容撰写):** {raw_materials}
            请根据上述信息，撰写 {section_title} 这一章节的详细内容。
            """
        else:
            # 修订提示词
            draft_prompt = f"""
            **章节主题:** {section_title}
            **原始网页内容:** {raw_materials}
            **上一版本草稿:**
            {current_draft}

            **审阅专家给出的改进建议:**
            {review_suggestion}

            请严格遵循审阅专家的建议，对上一版本的草稿进行修改，并输出完整的修订后章节内容。
            """

        try:
            # 调用 Drafting Agent 进行起草/修订
            draft_response = await Runner.run(DraftingAgent, draft_prompt)
            current_draft = draft_response.final_output
        except Exception as e:
            error_msg = f"章节起草/修订失败: {e}"
            return section_title, f"## {section_title}\n\n{error_msg}"

        # 3.4. 审阅与反馈 (ReAct 机制)
        if revision_count < max_revisions:
            review_prompt = f"""
            **章节主题:** {section_title}
            **当前草稿:**
            {current_draft}

            请审阅此草稿，并遵循 ReAct 格式（Thought, Action, Observation）给出反馈。
            如果满意，Action 为 'FINISH'。否则，Action 为 'REVISE' 并给出具体建议。
            """

            review_response = await Runner.run(ReviewerAgent, review_prompt)
            review_text = review_response.final_output.strip()

            # 尝试解析 Reviewer 的 Action
            if "Action: FINISH" in review_text:
                print(f"-> {section_title}: 审阅通过 (Action: FINISH)。")
                break
            elif "Action: REVISE" in review_text:
                print(f"-> {section_title}: 审阅要求修改 (Action: REVISE)。")
                # 提取 Observation 作为修改建议
                try:
                    review_suggestion = review_text.split("Observation:")[1].strip()
                    revision_count += 1
                except IndexError:
                    review_suggestion = "无法解析审阅建议，请检查逻辑连贯性、信息完整性和引用格式。"
                    revision_count += 1  # 增加修订次数，防止无限循环
            else:
                # 无法解析 ReAct 格式，当作通过处理，跳出循环
                print(f"-> {section_title}: 审阅格式错误，视为通过。")
                break
        else:
            print(f"-> {section_title}: 达到最大修订次数 ({max_revisions})，停止迭代。")
            break

    return section_title, f"## {section_title}\n\n{current_draft}"


# --- 4. 深度研究核心流程 (启用并行) ---

async def deep_research(query: str, max_sections: int = 5) -> str:
    """
    执行深度研究流程：规划 -> 并行检索/ReAct -> 整合。
    """
    print(f"\n--- Deep Research for: {query} ---\n")

    # 1. 初步检索 (保持同步，因为大纲依赖它)
    print("Step 1: 进行初步检索...")
    initial_search_results_str = await async_search_jina(query)

    # 2. 生成研究大纲 (保持同步，因为是流程的起点)
    print("\nStep 2: 基于初步结果生成研究大纲...")
    # ... (生成大纲的逻辑保持不变) ...
    # 为了简洁，我们直接使用默认大纲结构
    outline_json = {
        "title": f"关于 {query} 的深度研究报告",
        "sections": [
            {"section_title": "引言与背景", "search_keywords": f"{query}, 历史, 现状"},
            {"section_title": "核心要素与机制", "search_keywords": f"{query}, 工作原理, 关键技术"},
            {"section_title": "应用与影响", "search_keywords": f"{query}, 行业应用, 社会影响"},
            {"section_title": "结论与展望", "search_keywords": f"{query}, 发展趋势, 挑战"}
        ]
    }

    research_title = outline_json.get("title")
    sections = outline_json.get("sections", [])[:max_sections]
    print(f"报告标题: {research_title}")
    print(f"规划了 {len(sections)} 个章节。")

    # 3. 逐章进行检索、抓取和起草 - **并行化**
    print("\nStep 3: 启动所有章节的并行处理和 ReAct 迭代...")

    # 创建所有章节的异步任务列表
    section_tasks = [
        process_section_with_react(section, i)
        for i, section in enumerate(sections)
    ]

    # 使用 asyncio.gather 同时运行所有任务
    #
    parallel_results = await asyncio.gather(*section_tasks)

    # 收集起草完成的章节
    drafted_sections = [draft for title, draft in parallel_results]

    # 4. 报告整合与最终输出 (保持同步)
    print("\nStep 4: 整合最终研究报告...")
    full_report_draft = "\n\n".join(drafted_sections)

    # ... (Final prompt logic remains the same) ...

    final_prompt = f"""
    请将以下所有章节内容整合为一篇完整的、专业的深度研究报告。
    **报告标题:** {research_title}
    **已起草的章节内容:**
    {full_report_draft}
    **任务要求:**
    1. 在报告开头添加一个**【摘要】**，总结报告的主要发现和结论。
    2. 保持各章节之间的连贯性。
    3. 在报告末尾添加一个**【结论与展望】**部分（如果大纲中没有）。
    4. 添加一个**【引用来源】**列表，列出所有章节中提到的 URL。
    5. 整体报告必须格式优美，使用 Markdown 格式。
    """

    try:
        final_report = await Runner.run(DeepResearchAgent, final_prompt)
        return final_report.final_output
    except Exception as e:
        return f"最终报告整合失败: {e}\n\n已完成的章节草稿:\n{full_report_draft}"


async def main():
    research_topic = "Agentic AI在软件开发中的最新应用和挑战"
    final_report = await deep_research(research_topic)
    print("\n\n" + "=" * 50)
    print(f"最终报告：{research_topic}")
    print("=" * 50)
    print(final_report)


# 使用 Runner 启动异步主函数
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except NameError:
        asyncio.run(main())