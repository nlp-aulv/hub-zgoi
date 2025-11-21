import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server contains some api of news.""",
)

@mcp.tool
def get_today_daily_news():
    """Retrieves a list of today's daily news bulletin items from the external API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/bulletin?key={TOKEN}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_douyin_hot_news():
    """Retrieves a list of trending topics or hot news from Douyin (TikTok China) using the API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/douyinhot?key={TOKEN}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_github_hot_news():
    """Retrieves a list of trending repositories/projects on GitHub using the API."""
    try:
        return requests.get(f"https://whyta.cn/api/github?key={TOKEN}").json()["items"]
    except:
        return []

@mcp.tool
def get_toutiao_hot_news():
    """Retrieves a list of hot news headlines from Toutiao (a Chinese news platform) using the API."""
    try:
        print(f"https://whyta.cn/api/tx/topnews?key={TOKEN}")
        return requests.get(f"https://whyta.cn/api/tx/topnews?key={TOKEN}").json()["result"]["list"]
    except:
        import traceback
        traceback.print_exc()
        return []

@mcp.tool
def get_sports_news():
    """Retrieves a list of esports or general sports news items using the external API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/esports?key={TOKEN}").json()["result"]["newslist"]
    except:
        return []
