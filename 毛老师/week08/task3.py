from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from typing_extensions import Literal
from datetime import datetime
import openai
import json
import re
import uvicorn

# 初始化 OpenAI 客户端
client = openai.OpenAI(
    api_key="sk-78cc4e9ac8f44efdb207b7232e1ae8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 创建 FastAPI 应用
app = FastAPI(
    title="信息抽取智能体服务",
    description="提供基于函数调用和Prompt工程的信息抽取能力",
    version="1.0.0"
)


# 请求和响应模型
class ExtractionRequest(BaseModel):
    text: str = Field(..., description="需要抽取信息的文本")
    model_name: str = Field(default="qwen-plus", description="使用的模型名称")


class ExtractionResponse(BaseModel):
    domain: Optional[str] = None
    intent: Optional[str] = None
    entities: Dict[str, Any] = {}
    success: bool = True
    error: Optional[str] = None


# 信息抽取的数据模型
class IntentDomainNerTask(BaseModel):
    """对文本抽取领域类别、意图类型、实体标签"""
    domain: Literal[
        'music', 'app', 'radio', 'lottery', 'stock', 'novel', 'weather',
        'match', 'map', 'website', 'news', 'message', 'contacts', 'translation',
        'tvchannel', 'cinemas', 'cookbook', 'joke', 'riddle', 'telephone',
        'video', 'train', 'poetry', 'flight', 'epg', 'health', 'email', 'bus', 'story'
    ] = Field(description="用户提问的所属类型领域")
    intent: Literal[
        'OPEN', 'SEARCH', 'REPLAY_ALL', 'NUMBER_QUERY', 'DIAL', 'CLOSEPRICE_QUERY',
        'SEND', 'LAUNCH', 'PLAY', 'REPLY', 'RISERATE_QUERY', 'DOWNLOAD', 'QUERY',
        'LOOK_BACK', 'CREATE', 'FORWARD', 'DATE_QUERY', 'SENDCONTACTS', 'DEFAULT',
        'TRANSLATION', 'VIEW', 'NaN', 'ROUTE', 'POSITION'
    ] = Field(description="用户提问的意图")

    # 实体字段
    Src: Optional[str] = Field(default=None, description="来源")
    startDate_dateOrig: Optional[str] = Field(default=None, description="开始日期原始值")
    film: Optional[str] = Field(default=None, description="电影")
    endLoc_city: Optional[str] = Field(default=None, description="目的地城市")
    artistRole: Optional[str] = Field(default=None, description="艺术家角色")
    location_country: Optional[str] = Field(default=None, description="位置国家")
    location_area: Optional[str] = Field(default=None, description="位置区域")
    author: Optional[str] = Field(default=None, description="作者")
    startLoc_city: Optional[str] = Field(default=None, description="出发地城市")
    season: Optional[str] = Field(default=None, description="季节")
    dishNamet: Optional[str] = Field(default=None, description="菜品名称类型")
    media: Optional[str] = Field(default=None, description="媒体")
    datetime_date: Optional[datetime] = Field(default=None, description="日期时间-日期")
    episode: Optional[str] = Field(default=None, description="剧集")
    teleOperator: Optional[str] = Field(default=None, description="电信运营商")
    questionWord: Optional[str] = Field(default=None, description="疑问词")
    receiver: Optional[str] = Field(default=None, description="接收者")
    ingredient: Optional[str] = Field(default=None, description="食材")
    name: Optional[str] = Field(default=None, description="名称")
    startDate_time: Optional[datetime] = Field(default=None, description="开始时间")
    startDate_date: Optional[datetime] = Field(default=None, description="开始日期")
    location_province: Optional[str] = Field(default=None, description="位置省份")
    endLoc_poi: Optional[str] = Field(default=None, description="目的地兴趣点")
    artist: Optional[str] = Field(default=None, description="艺术家")
    dynasty: Optional[str] = Field(default=None, description="朝代")
    area: Optional[str] = Field(default=None, description="区域")
    location_poi: Optional[str] = Field(default=None, description="位置兴趣点")
    relIssue: Optional[str] = Field(default=None, description="相关问题")
    Dest: Optional[str] = Field(default=None, description="目的地")
    content: Optional[str] = Field(default=None, description="内容")
    keyword: Optional[str] = Field(default=None, description="关键词")
    target: Optional[str] = Field(default=None, description="目标")
    startLoc_area: Optional[str] = Field(default=None, description="出发地区域")
    tvchannel: Optional[str] = Field(default=None, description="电视频道")
    type: Optional[str] = Field(default=None, description="类型")
    song: Optional[str] = Field(default=None, description="歌曲")
    queryField: Optional[str] = Field(default=None, description="查询字段")
    awayName: Optional[str] = Field(default=None, description="客队名称")
    headNum: Optional[str] = Field(default=None, description="人数")
    homeName: Optional[str] = Field(default=None, description="主队名称")
    decade: Optional[str] = Field(default=None, description="年代")
    payment: Optional[str] = Field(default=None, description="支付方式")
    popularity: Optional[str] = Field(default=None, description="流行度")
    tag: Optional[str] = Field(default=None, description="标签")
    startLoc_poi: Optional[str] = Field(default=None, description="出发地兴趣点")
    date: Optional[str] = Field(default=None, description="日期")
    startLoc_province: Optional[str] = Field(default=None, description="出发地省份")
    endLoc_province: Optional[str] = Field(default=None, description="目的地省份")
    location_city: Optional[str] = Field(default=None, description="位置城市")
    absIssue: Optional[str] = Field(default=None, description="绝对问题")
    utensil: Optional[str] = Field(default=None, description="厨具")
    scoreDescr: Optional[str] = Field(default=None, description="分数描述")
    dishName: Optional[str] = Field(default=None, description="菜品名称")
    endLoc_area: Optional[str] = Field(default=None, description="目的地区域")
    resolution: Optional[str] = Field(default=None, description="分辨率")
    yesterday: Optional[str] = Field(default=None, description="昨天")
    timeDescr: Optional[str] = Field(default=None, description="时间描述")
    category: Optional[str] = Field(default=None, description="类别")
    subfocus: Optional[str] = Field(default=None, description="子焦点")
    theatre: Optional[str] = Field(default=None, description="剧院")
    datetime_time: Optional[datetime] = Field(default=None, description="日期时间-时间")


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
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
                        "required": response_model.model_json_schema().get('required', []),
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )

        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            result = response_model.model_validate_json(arguments)

            # 提取非空的实体字段
            entities = {}
            for field_name, field_value in result.dict().items():
                if field_name not in ['domain', 'intent'] and field_value is not None:
                    entities[field_name] = field_value

            return {
                "domain": result.domain,
                "intent": result.intent,
                "entities": entities,
                "success": True
            }
        except Exception as e:
            return {
                "domain": None,
                "intent": None,
                "entities": {},
                "success": False,
                "error": str(e)
            }


# API 路由
@app.get("/")
async def root():
    return {"message": "信息抽取智能体服务已启动", "status": "running"}


@app.post("/extract/function-call", response_model=ExtractionResponse)
async def extract_with_function_call(request: ExtractionRequest):
    """
    使用函数调用方式进行信息抽取
    """
    try:
        agent = ExtractionAgent(model_name=request.model_name)
        result = agent.call(request.text, IntentDomainNerTask)
        return ExtractionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"函数调用抽取失败: {str(e)}")


@app.post("/extract/prompt-engineering", response_model=ExtractionResponse)
async def extract_with_prompt_engineering(request: ExtractionRequest):
    """
    使用Prompt工程方式进行信息抽取
    """
    try:
        completion = client.chat.completions.create(
            model=request.model_name,
            messages=[
                {"role": "system", "content": """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
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
"""},
                {"role": "user", "content": request.text},
            ],
        )

        result = completion.choices[0].message.content

        # 使用正则表达式提取 JSON
        json_match = re.search(r'```json\s*(.*?)\s*```', result, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            return ExtractionResponse(**data)
        else:
            # 如果没有找到代码块，尝试直接解析整个响应
            try:
                data = json.loads(result)
                return ExtractionResponse(**data)
            except:
                return ExtractionResponse(
                    success=False,
                    error="无法解析模型响应"
                )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prompt工程抽取失败: {str(e)}")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

uvicorn.run(app, host="0.0.0.0", port=8000)

# http://localhost:8000/
# http://localhost:8000/health
"""
curl -X POST "http://localhost:8000/extract/function-call" \
     -H "Content-Type: application/json" \
     -d '{
         "text": "帮我查询下从北京到天津的在2025年10月1日的火车票",
         "model_name": "qwen-plus"
     }'
     
curl -X POST "http://localhost:8000/extract/prompt-engineering" \
     -H "Content-Type: application/json" \
     -d '{
         "text": "帮我查询下从北京到天津的在2025年10月1日的火车票",
         "model_name": "qwen-plus"
     }'
"""
