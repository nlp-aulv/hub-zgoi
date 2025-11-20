import asyncio
from fastmcp import FastMCP, Client
from typing import Optional, List

from news import mcp as news_mcp
from saying import mcp as saying_mcp
from tool import mcp as tool_mcp
from sentiment import mcp as sentiment_mcp

mcp = FastMCP(
    name="MCP-Server"
)

# 类别映射到对应的MCP服务器
CATEGORY_MCP_MAP = {
    "news": news_mcp,
    "saying": saying_mcp,
    "tool": tool_mcp,
    "sentiment": sentiment_mcp
}

CATEGORIES = list(CATEGORY_MCP_MAP.keys())

async def setup():
    """初始化设置，导入所有MCP服务器"""
    for category_mcp in CATEGORY_MCP_MAP.values():
        await mcp.import_server(category_mcp, prefix="")

@mcp.tool
def list_tools_by_category(categories: List[str] = None) -> dict:
    """
    根据指定类别列出可用工具。
    
    参数:
        categories: 工具类别列表。如果为None或空列表，则返回所有工具。
                  可选值: "news", "saying", "tool", "sentiment"
    
    返回:
        包含每个类别及其对应工具列表的字典
    """
    # 这个函数主要是为了展示API结构，在实际应用中，
    # 工具过滤将在服务器层面处理
    if not categories:
        categories = CATEGORIES
    
    result = {}
    for category in categories:
        if category in CATEGORY_MCP_MAP:
            # 在实际实现中，这里会查询特定类别的工具
            result[category] = f"Tools for {category} category"
    
    return result

@mcp.tool
def get_available_categories() -> List[str]:
    """
    获取所有可用的工具类别
    
    返回:
        所有可用类别的列表
    """
    return CATEGORIES

# 添加一个函数来根据类别创建动态MCP实例
def create_filtered_mcp(categories: List[str] = None):
    """
    根据指定类别创建一个过滤后的MCP实例
    
    参数:
        categories: 要包含的类别列表，如果为None则包含所有类别
        
    返回:
        配置好的FastMCP实例
    """
    filtered_mcp = FastMCP(name="Filtered-MCP-Server")
    
    # 如果没有指定类别，则使用所有类别
    if not categories:
        categories = CATEGORIES
        
    # 只导入指定类别的MCP服务器
    for category in categories:
        if category in CATEGORY_MCP_MAP:
            # 注意：实际实现可能需要不同的方法来实现动态导入
            # 这里只是一个概念演示
            pass
            
    return filtered_mcp

async def test_filtering():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        print("可用工具:", [t.name for t in tools])

if __name__ == "__main__":
    asyncio.run(setup())
    asyncio.run(test_filtering())
    mcp.run(transport="sse", port=8900)