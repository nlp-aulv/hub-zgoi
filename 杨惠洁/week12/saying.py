import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Saying-MCP-Server",
    instructions="""This server contains some api of saying.""",
)

@mcp.tool
def get_today_familous_saying():
    """Retrieves a random famous saying or 'hitokoto' quote using the external API."""
    try:
        return requests.get(f"https://whyta.cn/api/yiyan?key={TOKEN}").json()["hitokoto"]
    except:
        return []

@mcp.tool
def get_today_motivation_saying():
    """Retrieves a motivation saying or inspirational quote from the API."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/lzmy?key={TOKEN}").json()["result"]
    except:
        return []

@mcp.tool
def get_today_working_saying():
    """Retrieves a quote related to work or chicken soup for the soul (心灵鸡汤) content."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/lzmy?key={TOKEN}").json()["result"]["content"]
    except:
        return []