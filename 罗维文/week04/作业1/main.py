# python自带库
import time
import traceback
from typing import Union, Dict, List, Any, Union, Optional

# 第三方库
# import openai
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 自己写的模块
# from data_schema import TextClassifyResponse
# from data_schema import TextClassifyRequest
from bert import model_for_bert
from logger import logger


class TextClassifyRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")

class TextClassifyResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")

app = FastAPI()

@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用bert进行文本分类

    :param req: 请求体
    """
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    # info 日志
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
