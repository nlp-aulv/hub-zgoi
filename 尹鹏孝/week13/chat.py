import requests
TOKEN = "738b541a5f7a"
from agents import Agent, InputGuardrail, GuardrailFunctionOutput, Runner
from fastmcp import FastMCP
mcp = FastMCP(
    name="Chat-MCP-Server",
    instructions="""This server contains some api of AI Chat.""",
)

"""
[Tool(name='Chat AI', title=None, description="Chat bulletin items from the external API.", inputSchema={'properties': {}, 'type': 'object'}, outputSchema=None, icons=None, annotations=None, meta={'_fastmcp': {'tags': []}})
"""
chat_agent = Agent(
    name="chat AI",
    model="qwen-max",
    handoff_description="负责聊天的大咖。",
    instructions="您是心理咨询师。请给用户以轻松愉悦积极健康的聊天内容和互动。",
)
@mcp.tool
async def chat_ai():
    """Retrieves a list of today's daily news bulletin items from the external API."""
    try:
        # return requests.get(f"https://whyta.cn/api/tx/bulletin?key={TOKEN}", timeout=5).json()["result"]["list"]

        query = "AI闲聊助手？"
        print(f"**用户提问:** {query}")
        result = await Runner.run(chat_agent, query)
        return result
    except:
        return ''



