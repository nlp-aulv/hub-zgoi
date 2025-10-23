
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import Literal as LiteralExt
import uvicorn

from Classify import classify_domain, classify_intent, classify_slots

# 创建FastAPI应用
app = FastAPI(
    title="文本分类API",
    description="提供意图识别、领域识别和实体识别三个接口",
    version="1.0.0"
)

# 读取领域类别文件
with open('domains.txt', 'r', encoding='utf-8') as f:
    domains = [line.strip() for line in f.readlines()]

# 读取意图类别文件
with open('intents.txt', 'r', encoding='utf-8') as f:
    intents = [line.strip() for line in f.readlines()]

# 读取实体类别文件
with open('slots.txt', 'r', encoding='utf-8') as f:
    slots = [line.strip() for line in f.readlines()]

# 请求模型
class TextRequest(BaseModel):
    text: str = Field(..., description="需要进行分类的文本", example="我想听周杰伦的歌")
    model_name: str = Field("qwen-plus", description="使用的模型名称")

# 响应模型
class DomainResponse(BaseModel):
    domain: str = Field(..., description="识别出的领域类别")

class IntentResponse(BaseModel):
    intent: str = Field(..., description="识别出的意图类别")

class SlotsResponse(BaseModel):
    slots: List[str] = Field(..., description="识别出的实体类别列表")

# 领域识别接口
@app.post("/classify/domain", response_model=DomainResponse)
async def classify_domain_endpoint(request: TextRequest):
    """
    领域识别接口
    - **text**: 需要进行分类的文本
    - **model_name**: 使用的模型名称 (默认为qwen-plus)
    """
    result = classify_domain(request.text, request.model_name)
    return {"domain": result}

# 意图识别接口
@app.post("/classify/intent", response_model=IntentResponse)
async def classify_intent_endpoint(request: TextRequest):
    """
    意图识别接口
    - **text**: 需要进行分类的文本
    - **model_name**: 使用的模型名称 (默认为qwen-plus)
    """
    result = classify_intent(request.text, request.model_name)
    return {"intent": result}

# 实体识别接口
@app.post("/classify/slots", response_model=SlotsResponse)
async def classify_slots_endpoint(request: TextRequest):
    """
    实体识别接口
    - **text**: 需要进行分类的文本
    - **model_name**: 使用的模型名称 (默认为qwen-plus)
    """
    result = classify_slots(request.text, request.model_name)
    return {"slots": result}

# 根路径
@app.get("/")
async def root():
    return {
        "message": "欢迎使用文本分类API",
        "routes": {
            "领域识别": "/classify/domain",
            "意图识别": "/classify/intent",
            "实体识别": "/classify/slots"
        }
    }

# 启动命令
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
