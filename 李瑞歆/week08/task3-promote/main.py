from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional, List

# 导入意图识别模块
from intent_detection import detect_intent, IntentDetectionResponse

from extraction_service import (
    extraction_service,
    IntentRecognitionRequest,
    IntentRecognitionResponse,
    DomainRecognitionRequest,
    DomainRecognitionResponse,
    EntityRecognitionRequest,
    EntityRecognitionResponse
)

# 创建 FastAPI 应用实例
app = FastAPI(
    title="意图识别API",
    description="一个基于 FastAPI 的意图识别服务。",
    version="1.0.0"
)

# 请求数据模型
class IntentRequest(BaseModel):
    """
    意图识别请求模型
    """
    text: str

    class Config:
        schema_extra = {
            "example": {
                "text": "给庄小雷发短信"
            }
        }

@app.post("/intent/detect",
          response_model=IntentDetectionResponse,
          summary="意图识别",
          description="对输入文本进行意图识别，返回意图类型、领域和实体信息。")
async def intent_detection(request: IntentRequest):
    """
    对输入文本进行意图识别。

    - **text**: 需要识别的文本内容
    """
    try:
        # 调用意图识别函数
        result = detect_intent(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

class ErrorResponse(BaseModel):
    """错误响应模型"""
    error: str
    message: str

@app.post("/intent/recognize",
          response_model=IntentRecognitionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          summary="意图识别",
          description="识别电商场景下的用户意图，包括业务领域、具体意图和商品类别")
async def recognize_intent(request: IntentRecognitionRequest):
    """
    意图识别接口

    - **text**: 需要识别意图的文本，如"我想买一台苹果手机，最新款的，有什么优惠吗？"
    """
    try:
        result = extraction_service.recognize_intent(request.text)
        if result is None:
            raise HTTPException(status_code=500, detail="意图识别失败，请稍后重试")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/domain/recognize",
          response_model=DomainRecognitionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          summary="领域识别",
          description="识别学术研究领域，包括学科门类、研究子领域和研究类型")
async def recognize_domain(request: DomainRecognitionRequest):
    """
    领域识别接口

    - **text**: 需要识别领域的文本，如"基于深度学习的图像识别在医疗诊断中的应用研究"
    """
    try:
        result = extraction_service.recognize_domain(request.text)
        if result is None:
            raise HTTPException(status_code=500, detail="领域识别失败，请稍后重试")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")

@app.post("/entity/recognize",
          response_model=EntityRecognitionResponse,
          responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
          summary="实体识别",
          description="识别文本中的实体信息，包括人名、组织机构、地名、时间表达式等")
async def recognize_entities(request: EntityRecognitionRequest):
    """
    实体识别接口

    - **text**: 需要识别实体的文本，如"昨天马云在阿里巴巴杭州总部会见了腾讯公司的马化腾"
    """
    try:
        result = extraction_service.recognize_entities(request.text)
        if result is None:
            raise HTTPException(status_code=500, detail="实体识别失败，请稍后重试")
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")




# 运行应用
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)