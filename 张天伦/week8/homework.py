from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Optional, List
import openai
import uvicorn

# 客户端配置
client = openai.OpenAI(
    api_key="sk-78cc4e9ac8f44efdb207b7232ed8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


# 定义请求和响应模型
class ExtractionRequest(BaseModel):
    text: str = Field(description="需要抽取信息的文本")
    model_name: Optional[str] = Field(default="qwen-plus", description="模型名称")


class SlotItem(BaseModel):
    entity_type: str = Field(description="实体类型")
    entity_value: str = Field(description="实体值")


class ExtractionResponse(BaseModel):
    domain: str = Field(description="领域类别")
    intent: str = Field(description="意图类型")
    slots: Dict[str, str] = Field(description="实体识别结果")
    success: bool = Field(description="是否成功")
    message: Optional[str] = Field(default="", description="额外信息")


class BatchExtractionRequest(BaseModel):
    texts: List[str] = Field(description="批量处理的文本列表")
    model_name: Optional[str] = Field(default="qwen-plus", description="模型名称")


class BatchExtractionResponse(BaseModel):
    results: List[ExtractionResponse] = Field(description="批量处理结果")
    total: int = Field(description="总处理数量")
    success_count: int = Field(description="成功处理数量")


# 信息抽取工具类
class ExtractionAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def call(self, user_prompt: str):
        """调用模型进行信息抽取"""

        # 系统提示词
        system_prompt = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

请准确识别文本的领域、意图和实体信息。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        try:
            # 解析响应内容
            content = response.choices[0].message.content
            # 这里需要根据实际的响应格式进行解析
            # 假设响应是 JSON 格式
            import json
            import re

            # 尝试从响应中提取 JSON
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                result_data = json.loads(json_str)

                return ExtractionResponse(
                    domain=result_data.get("domain", ""),
                    intent=result_data.get("intent", ""),
                    slots=result_data.get("slots", {}),
                    success=True,
                    message="抽取成功"
                )
            else:
                return ExtractionResponse(
                    domain="",
                    intent="",
                    slots={},
                    success=False,
                    message="无法解析模型响应"
                )

        except Exception as e:
            return ExtractionResponse(
                domain="",
                intent="",
                slots={},
                success=False,
                message=f"处理错误: {str(e)}"
            )

    def extract(self, text: str) -> ExtractionResponse:
        """提取文本的领域、意图和实体信息"""
        return self.call(text)


# 创建 FastAPI 应用
app = FastAPI(
    title="信息抽取API",
    description="基于大模型的意图识别、领域识别和实体识别服务",
    version="1.0.0"
)

# 全局代理实例
extraction_agent = ExtractionAgent()


@app.get("/")
async def root():
    return {"message": "信息抽取API服务已启动", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "extraction-api"}


@app.post("/extract", response_model=ExtractionResponse)
async def extract_info(request: ExtractionRequest):
    """
    单文本信息抽取接口
    """
    try:
        # 如果指定了模型名称，创建新的代理实例
        if request.model_name != "qwen-plus":
            agent = ExtractionAgent(request.model_name)
        else:
            agent = extraction_agent

        result = agent.extract(request.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.post("/batch_extract", response_model=BatchExtractionResponse)
async def batch_extract_info(request: BatchExtractionRequest):
    """
    批量文本信息抽取接口
    """
    try:
        results = []
        success_count = 0

        for text in request.texts:
            if request.model_name != "qwen-plus":
                agent = ExtractionAgent(request.model_name)
            else:
                agent = extraction_agent

            result = agent.extract(text)
            results.append(result)
            if result.success:
                success_count += 1

        return BatchExtractionResponse(
            results=results,
            total=len(request.texts),
            success_count=success_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"批量处理失败: {str(e)}")


@app.get("/domains")
async def get_available_domains():
    """
    获取可用的领域类别列表
    """
    domains = [
        "music", "app", "radio", "lottery", "stock", "novel", "weather",
        "match", "map", "website", "news", "message", "contacts",
        "translation", "tvchannel", "cinemas", "cookbook", "joke",
        "riddle", "telephone", "video", "train", "poetry", "flight",
        "epg", "health", "email", "bus", "story"
    ]
    return {"domains": domains}


@app.get("/intents")
async def get_available_intents():
    """
    获取可用的意图类别列表
    """
    intents = [
        "OPEN", "SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL",
        "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY", "REPLY",
        "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE",
        "FORWARD", "DATE_QUERY", "SENDCONTACTS", "DEFAULT",
        "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION"
    ]
    return {"intents": intents}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
