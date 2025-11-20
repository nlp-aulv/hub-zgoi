import requests

TOKEN = "https://ltpapi.xfyun.cn/v2/sa"

from fastmcp import FastMCP

mcp = FastMCP(
    name="Sentiment analysis service",
    instructions="""This server contains some api of Sentiment analysis.""",
)

@mcp.tool
def sentiment_analysis(text):
    url = f"{TOKEN}"
    data = {"text": text}
    response = requests.post(url, json=data)
    return response.json()
