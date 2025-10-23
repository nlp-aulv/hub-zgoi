from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, List, Union, Optional
import time
import openai
import traceback
from zy3_prompt import model_for_prompt
from zy3_tools import model_for_tools
app = FastAPI()


# =================== API 请求/响应模型 ===================
class PredictRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


class PredictItem(BaseModel):
    text_error: Optional[str] = Field(..., description="文本异常信息")
    domain: str = Field(..., description="领域")
    intent: str = Field(..., description="意图")
    slots: Dict[str, List[str]] = Field(..., description="实体")


class PredictResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: List[PredictItem] = Field(..., description="识别结果")
    classify_time: float = Field(..., description="识别耗时")
    error_msg: str = Field(..., description="响应异常信息")


# 例子
# {
#   "request_id": "string",
#   "request_text": ["srfsksfh四jil","帮我查询下从北京到天津到武汉的汽车票","帮我播放周杰伦的歌曲",""]
# }

@app.post("/v1/text-cls/prompt")
def prompt_classify(req: PredictRequest) -> PredictResponse:
    start_time = time.time()
    response = PredictResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result=[],
        classify_time=0,
        error_msg=""
    )
    try:
        response.classify_result = model_for_prompt(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = []
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response

@app.post("/v1/text-cls/tools")
def tools_classify(req: PredictRequest) -> PredictResponse:
    start_time = time.time()
    response = PredictResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result=[],
        classify_time=0,
        error_msg=""
    )
    try:
        response.classify_result = model_for_tools(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = []
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response

