import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server contains some api of news.""",
)

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