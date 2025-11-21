import asyncio
from fastmcp import FastMCP, Client
from typing import Optional, List

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from sentiment import mcp as sentiment_mcp

# 类别映射到对应的MCP服务器
CATEGORY_MCP_MAP = {
    "news": news_mcp,
    "saying": saying_mcp,
    "tool": tool_mcp,
    "sentiment": sentiment_mcp
}

CATEGORIES = list(CATEGORY_MCP_MAP.keys())

# 创建主MCP实例
mcp = FastMCP(
    name="MCP-Server"
)

# 为单个类别创建专门的MCP实例
news_only_mcp = FastMCP(name="News-Only-MCP-Server")
saying_only_mcp = FastMCP(name="Saying-Only-MCP-Server")
tool_only_mcp = FastMCP(name="Tool-Only-MCP-Server")
sentiment_only_mcp = FastMCP(name="Sentiment-Only-MCP-Server")

# 专门MCP实例映射
SERVER_MCP_MAP = {
    "news_only_mcp": news_only_mcp,
    "saying_only_mcp": saying_only_mcp,
    "tool_only_mcp": tool_only_mcp,
    "sentiment_only_mcp": sentiment_only_mcp
}

async def setup():
    """初始化设置，导入所有MCP服务器"""
    # 设置主MCP服务器
    for category_mcp in CATEGORY_MCP_MAP.values():
        await mcp.import_server(category_mcp, prefix="")
    
    # 设置专门的MCP服务器
    await news_only_mcp.import_server(news_mcp, prefix="")
    await saying_only_mcp.import_server(saying_mcp, prefix="")
    await tool_only_mcp.import_server(tool_mcp, prefix="")
    await sentiment_only_mcp.import_server(sentiment_mcp, prefix="")

@mcp.tool
def get_available_categories() -> List[str]:
    """
    获取所有可用的工具类别
    
    返回:
        所有可用类别的列表
    """
    return CATEGORIES

# 添加一个工具函数来获取过滤后的工具列表
@mcp.tool
def get_filtered_tools(categories: str = None) -> dict:
    """
    根据类别参数返回过滤后的工具
    
    参数:
        categories: 逗号分隔的类别字符串，例如 "news,tool"
        
    返回:
        过滤后的工具列表
    """
    if categories:
        category_list = categories.split(",")
        # 验证类别
        valid_categories = [cat for cat in category_list if cat in CATEGORY_MCP_MAP]
    else:
        valid_categories = CATEGORIES
    
    # 返回工具信息
    return {
        "tools": [f"tool_from_{cat}_category" for cat in valid_categories],
        "categories": valid_categories,
        "message": "This is a simulated response. In a full implementation, this would return actual tools."
    }

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("可用工具:", [t.name for t in tools])

if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)