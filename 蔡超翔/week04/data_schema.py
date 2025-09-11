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
    request_id: str
    request_text: str
    classify_result: list  # 或 List[float] 如果知道具体维度
    classify_time: float
    error_msg: str = ""
