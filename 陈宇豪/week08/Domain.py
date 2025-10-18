from typing import Dict, Any, Optional

from pydantic import Field
from pydantic import BaseModel


class Request(BaseModel):
    """!!! abstract "Usage Documentation"""""
    request_user: Optional[str] = Field(..., description="用于获取历史对话")
    request_text: str = Field(..., description="询问问题")


class Response(BaseModel):
    request_user: Optional[str] = Field(..., description="用于获取历史对话")
    response_text: str = Field(..., description="大模型回答")
    error_msg: str = Field(..., description="异常信息")
