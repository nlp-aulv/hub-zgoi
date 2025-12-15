from typing import Annotated, Union
import requests

TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Emotion-MCP-Server",
    instructions="""This server contains some api of Emotion.""",
)

@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze sentiment"]):
    """Classifies the sentiment of a given text (positive, negative, neutral)."""
    print("------sentiment_classification-------")
    try:
        response = requests.get(f"https://whyta.cn/api/tx/nlpchat?key={TOKEN}&word={text}")
        data = response.json()
        print(data)
        return data.get("result", {})
    except Exception as e:
        return {"error": f"Sentiment analysis failed: {str(e)}"}