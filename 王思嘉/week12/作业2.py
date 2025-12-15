import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="News-MCP-Server",
    instructions="""This server contains some api of news.""",
)

@mcp.tool
def get_sentiment_classification():
    
    try:
        return requests.get(f"https://whyta.cn/api/tx/sentiment?key={TOKEN}").json()["result"]["list"]
    except:
        return []
