from typing import Annotated

import requests
TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Sentiment-MCP-Server",
    instructions="""This server contains some api of sentimen.""",
)



@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""

    try:
        response = requests.get(f"https://127.0.0.1:8000/sentiment?key={TOKEN}&text={text}")

        if response.status_code == 200:
            return response.json()
        else:
            return []

    except Exception as e:

        return f"连接失败：{str(e)}，请确保API服务已启动"