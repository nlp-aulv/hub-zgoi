import openai
from fastapi import FastAPI, HTTPException
import requests
from pydantic import BaseModel, Field
import openai
import json
from typing import Dict, Any, Optional
from typing_extensions import Literal

# 初始化
app = FastAPI(
    title="智能信息抽取",
    description="基于OpenAI的智能信息抽取",
    version="0.1.0",
)

client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-cc43ca2821f64bfa9c6e20bf0889d92c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 请求数据模型
class ExtractionRequest(BaseModel):
    text: str
    model: str = "qwen-plus"

# 响应数据模型
class ExtractionResponse(BaseModel):
    domain: str
    intent: str
    slots: Dict[str, str]
    success: bool
    message: str = ""

class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal['music', 'app', 'weather', 'bus'] = Field(description="领域")
    intent: Literal['OPEN', 'SEARCH', 'QUERY'] = Field(description="意图")
    slots: Dict[str, str] = Field(default_factory=dict, description="实体槽位")

# 定义只能抽取的agent带tools
class ExtractionAgent:
    def __init__(self, model_name:str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt,
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                    },
                },
            }
        ]
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # 输出response
        print(response)
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except Exception as e:
            print(e)
            return None

# 定义提示词
prompt = """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：music / app / radio / lottery / stock / novel / weather / match / map / website / news / message / contacts / translation / tvchannel / cinemas / cookbook / joke / riddle / telephone / video / train / poetry / flight / epg / health / email / bus / story
- 待选的意图类别：OPEN / SEARCH / REPLAY_ALL / NUMBER_QUERY / DIAL / CLOSEPRICE_QUERY / SEND / LAUNCH / PLAY / REPLY / RISERATE_QUERY / DOWNLOAD / QUERY / LOOK_BACK / CREATE / FORWARD / DATE_QUERY / SENDCONTACTS / DEFAULT / TRANSLATION / VIEW / NaN / ROUTE / POSITION
- 待选的实体标签：code / Src / startDate_dateOrig / film / endLoc_city / artistRole / location_country / location_area / author / startLoc_city / season / dishNamet / media / datetime_date / episode / teleOperator / questionWord / receiver / ingredient / name / startDate_time / startDate_date / location_province / endLoc_poi / artist / dynasty / area / location_poi / relIssue / Dest / content / keyword / target / startLoc_area / tvchannel / type / song / queryField / awayName / headNum / homeName / decade / payment / popularity / tag / startLoc_poi / date / startLoc_province / endLoc_province / location_city / absIssue / utensil / scoreDescr / dishName / endLoc_area / resolution / yesterday / timeDescr / category / subfocus / theatre / datetime_time

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

```json
{
    "domain": ,
    "intent": ,
    "slots": {
      "待选实体": "实体名词",
    }
}
```
"""

@app.get("/")
async def root():
    return {"message": "智能信息抽取API服务"}

# 利用提示词
@app.post("/promptExtract", response_model=ExtractionResponse, tags=["信息抽取"])
async def extract_prompt(request: ExtractionRequest):
    try:
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt,
                },
                {
                    "role": "user",
                    "content": request.text,
                },
            ],
        )

        # 解析模型的返回结果
        result_content = response.choices[0].message.content

        # 提取JSON部分
        json_start = result_content.find('{')
        json_end = result_content.rfind('}') + 1
        json_str = result_content[json_start:json_end]

        # 解析JSON
        result_data = json.loads(json_str)

        return ExtractionResponse(
            domain=result_data["domain"],
            intent=result_data["intent"],
            slots=result_data["slots"],
            success=True,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"信息抽取失败: {str(e)}"
        )


# 利用tools
@app.post("/toolsExtract", response_model=IntentDomainNerTask, tags=["信息抽取"])
async def extract_tools(request: ExtractionRequest):
    return ExtractionAgent("qwen-plus").call(request.text, IntentDomainNerTask)

# 利用coze
@app.post("/cozeExtract", response_model=str, tags=["信息抽取"])
async def extract_coze(request: ExtractionRequest):
    url = 'https://api.coze.cn/open_api/v2/chat'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'pat_PKcYmUqCx7AVIjl2DhzStjog1WoELuqiWKQjwIDXDkrtNqFoHaBf9uT63h33bds5',
        'Accept': 'application/json',
        'Host': 'api.coze.cn',
        'Connection': 'keep-alive',
    }

    data = {
        "bot_id": "7562035397669765160",
        "user": "11",
        "query": request.text,
        "stream": False,
    }

    # 将宁典转换为JSON宁符串
    json_data = json.dumps(data)
    # 发送POST请求
    response = requests.post(url, headers = headers, data = json_data)
    return response.text