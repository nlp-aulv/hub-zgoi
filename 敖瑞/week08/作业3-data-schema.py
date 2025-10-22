from pydantic import BaseModel, Field
from typing import List, Union, Optional


class TextIdentificationRequest(BaseModel):
    """
    请求格式
    """
    request_id: Optional[str] = Field(..., description='请求id，方便调试')
    request_text: Union[str, List[str]] = Field(..., description='请求文本、字符串或列表')


class TextIdentificationResponse(BaseModel):
    """
    接口返回格式
    """
    request_id: Optional[str] = Field(..., description='请求id')
    request_text: Union[str, List[str]] = Field(..., description='请求文本、字符串或列表')
    identification_result: Union[str, List[str]] = Field(..., description='提取结果')
    identification_time: float = Field(..., description='提取耗时')
    error_msg: str = Field(..., description='异常信息')
