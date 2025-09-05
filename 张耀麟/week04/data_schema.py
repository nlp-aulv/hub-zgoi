from pydantic import BaseModel, Field
from typing import Dict, List, Any, Union, Optional


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
