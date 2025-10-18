from typing import Dict, Any, Optional

from pydantic import Field
from pydantic import BaseModel


class Request(BaseModel):
    """!!! abstract "Usage Documentation"""""
    request_user: Optional[str] = Field(..., description="用于获取历史对话")
    request_text: str = Field(..., description="询问问题")


class Response(BaseModel):
    request_user: Optional[str] = Field(..., description="用于获取历史对话")
    entities: Dict[str, str] = Field(..., description="抽取实体")
    domain: str = Field(..., description="抽取领域")
    intent: str = Field(..., description="抽取意图")
    error_msg: Optional[str] = Field(..., description="异常信息")
