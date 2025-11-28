import asyncio
from fastmcp import Client

client = Client("http://localhost:8900/sse")

async def call_tool():
    # async with client:
    #     result = await client.call_tool("get_today_daily_news")
    #     print(result)

    async with client:
        result = await client.list_tools()
        print(result)

        result = await client.call_tool("get_today_motivation_saying")
        print(result)

asyncio.run(call_tool())
