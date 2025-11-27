"""
https://www.autostock.cn/#/trade/stock
https://s.apifox.cn/c3278b4f-5629-4732-858c-36758ff5d083/api-147275957
"""
TOKEN = "zgaLG8unUPr"

import requests  # type: ignore
from typing import Annotated
from typing import Optional, Dict
import traceback
from fastapi import FastAPI, APIRouter  # type: ignore

app = FastAPI(
    name="Stock api Server",
    instructions="""This server provides stock basic tools.""",
)

# path get_stock_code http服务的路径
# operation_id get_stock_code mcp服务的名字
@app.get("/get_stock_code", operation_id="get_stock_codes")
async def get_all_stock_code(
        keyword: Annotated[Optional[str], "支持代码和名称模糊查询"] = None
) -> Dict:
    """所有股票，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/all" + "?token=" + TOKEN
    if keyword:
        url += "&keyWord=" + keyword

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_index_code", operation_id="get_index_code")
async def get_all_index_code():
    """所有指数，支持代码和名称模糊查询"""
    url = "https://api.autostock.cn/v1/stock/index/all" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_industry_code", operation_id="get_industry_code")
async def get_stock_industry_code():
    """获取板块数据"""
    url = "https://api.autostock.cn/v1/stock/industry/rank" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_board_info", operation_id="get_board_info")
async def get_stock_board_info():
    """获取大盘数据"""
    url = "https://api.autostock.cn/v1/stock/board" + "?token=" + TOKEN
    payload = {}
    headers = {}

    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_rank", operation_id="get_stock_rank")
async def get_stock_rank(
        node: Annotated[str, "股票市场/板块代码: {'a','b','ash','asz','bsh','bsz'} a(沪深A股)"],
        industryCode: Annotated[Optional[str], "行业代码，可选"] = None,
        pageIndex: Annotated[int, "页码"] = 1,
        pageSize: Annotated[int, "每页大小"] = 100,
        sort: Annotated[str, "排序字段: price,priceChange,pricePercent,buy,sell,open,close,high,low,volume,turnover 默认price(交易价格)。"] = "price",
        asc: Annotated[int, "排序方式: 0=降序(默认), 1=升序"] = 0
) -> Dict:
    """股票排行"""
    url = "https://api.autostock.cn/v1/stock/rank" + "?token=" + TOKEN
    headers = {}  # type: ignore

    try:
        payload = {
            "node": node,
            "industryCode": industryCode,
            "pageIndex": pageIndex,
            "pageSize": pageSize,
            "sort": sort,
            "asc": asc
        }
        response = requests.request("POST", url, headers=headers, json=payload, timeout=5)
        return response.json()
    except Exception as e:
        print(traceback.format_exc())
        return {}

@app.get("/get_month_line", operation_id="get_month_line")
async def get_stock_month_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """月k"""
    url = "https://api.autostock.cn/v1/stock/kline/month" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_week_line", operation_id="get_week_line")
async def get_stock_week_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
):
    """周k"""
    url = "https://api.autostock.cn/v1/stock/kline/week" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_day_line", operation_id="get_day_line")
async def get_stock_day_kline(
        code: Annotated[str, "股票代码"],
        startDate: Annotated[Optional[str], "开始时间(非必填)"] = None,
        endDate: Annotated[Optional[str], "结束时间(非必填)"] = None,
        type: Annotated[int, "0不复权,1前复权,2后复权"] = 0
) -> Dict:
    """日k"""
    url = "https://api.autostock.cn/v1/stock/kline/day" + "?token=" + TOKEN

    headers = {}  # type: ignore
    try:
        payload = {
            "code": code,
            "startDate": startDate,
            "endDate": endDate,
            "type": type
        }
        response = requests.request("GET", url, headers=headers, params=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_info", operation_id="get_stock_info")
async def get_stock_info(code: Annotated[str, "股票代码"]) -> Dict:
    """股票基础信息"""
    url = "https://api.autostock.cn/v1/stock" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}

@app.get("/get_stock_minute_data", operation_id="get_stock_minute_data")
async def get_stock_minute_data(code: str):
    """分时信息"""
    url = "https://api.autostock.cn/v1/stock/min" + "?token=" + TOKEN + "&code=" + code

    payload = {}  # type: ignore
    headers = {}  # type: ignore
    try:
        response = requests.request("GET", url, headers=headers, data=payload, timeout=10)
        return response.json()
    except Exception:
        print(traceback.format_exc())
        return {}
