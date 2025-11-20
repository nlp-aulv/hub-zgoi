from typing import Annotated, Union
import requests

# 导入 transformers 库用于情感分析
from transformers import pipeline

TOKEN = "738b541a5f7a"

from fastmcp import FastMCP
mcp = FastMCP(
    name="Tools-MCP-Server",
    instructions="""This server contains some api of tools.""",
)

# 初始化情感分析模型
try:
    # 尝试加载一个支持中文的情感分析模型
    sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-multilingual-cased-sentiment", tokenizer="distilbert-base-multilingual-cased-sentiment")
except Exception as e:
    print(f"情感分析模型加载失败，将使用默认简易分类器: {e}")
    sentiment_analyzer = None


@mcp.tool
def get_city_weather(city_name: Annotated[str, "The Pinyin of the city name (e.g., 'beijing' or 'shanghai')"]):
    """Retrieves the current weather data using the city's Pinyin name."""
    try:
        return requests.get(f"https://whyta.cn/api/tianqi?key={TOKEN}&city={city_name}").json()["data"]
    except:
        return []

@mcp.tool
def get_address_detail(address_text: Annotated[str, "City Name"]):
    """Parses a raw address string to extract detailed components (province, city, district, etc.)."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/addressparse?key={TOKEN}&text={address_text}").json()["result"]
    except:
        return []

@mcp.tool
def get_tel_info(tel_no: Annotated[str, "Tel phone number"]):
    """Retrieves basic information (location, carrier) for a given telephone number."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/mobilelocal?key={TOKEN}&phone={tel_no}").json()["result"]
    except:
        return []

@mcp.tool
def get_scenic_info(scenic_name: Annotated[str, "Scenic/tourist place name"]):
    """Searches for and retrieves information about a specific scenic spot or tourist attraction."""
    # https://apis.whyta.cn/docs/tx-scenic.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/scenic?key={TOKEN}&word={scenic_name}").json()["result"]["list"]
    except:
        return []

@mcp.tool
def get_flower_info(flower_name: Annotated[str, "Flower name"]):
    """Retrieves the flower language (花语) and details for a given flower name."""
    # https://apis.whyta.cn/docs/tx-huayu.html
    try:
        return requests.get(f"https://whyta.cn/api/tx/huayu?key={TOKEN}&word={flower_name}").json()["result"]
    except:
        return []

@mcp.tool
def get_rate_transform(
    source_coin: Annotated[str, "The three-letter code (e.g., USD, CNY) for the source currency."], 
    aim_coin: Annotated[str, "The three-letter code (e.g., EUR, JPY) for the target currency."], 
    money: Annotated[Union[int, float], "The amount of money to convert."]
):
    """Calculates the currency exchange conversion amount between two specified coins."""
    try:
        return requests.get(f"https://whyta.cn/api/tx/fxrate?key={TOKEN}&fromcoin={source_coin}&tocoin={aim_coin}&money={money}").json()["result"]["money"]
    except:
        return []

# 移除 get_news 工具的定义，因为它现在在 news.py 中定义
# @mcp.tool
# def get_news(query: Annotated[str, "新闻查询关键词"]):
#     """
#     # ... 模拟新闻查询逻辑 ...
#     """

@mcp.tool
def sentiment_classification(text: Annotated[str, "要分析情感的文本"]) -> Annotated[str, "文本的情感类别，例如 '积极', '消极', '中性'"]:
    """对给定文本进行情感分类。"""
    if sentiment_analyzer:
        result = sentiment_analyzer(text)[0]
        label = result['label']
        score = result['score']
        # 简单的映射，可以根据模型输出调整
        if "pos" in label.lower():
            return f"积极 (置信度: {score:.2f})"
        elif "neg" in label.lower():
            return f"消极 (置信度: {score:.2f})"
        else:
            return f"中性 (置信度: {score:.2f})"
    else:
        # 简易分类器作为备用
        if "好" in text or "喜欢" in text or "棒" in text:
            return "积极 (简易分类)"
        elif "差" in text or "不喜欢" in text or "糟糕" in text:
            return "消极 (简易分类)"
        else:
            return "中性 (简易分类)"

