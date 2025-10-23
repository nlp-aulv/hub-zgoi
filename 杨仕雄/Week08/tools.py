from pydantic import BaseModel, Field
from typing import List,Optional
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-cb217fdebe8c46dbb7e5e6aa76f9da89", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


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
                    "name": response_model.model_json_schema()['title'], # 一个内置方法，用来把模型的结构信息导出为 标准的 JSON Schema。
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
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
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class TextDomainEntity(BaseModel):
    """对文本进行意图识别、领域识别、实体识别"""
    domain: Literal['music','app','radio','lottery','stock','novel','weather','match','map','website','news','message','contacts','translation','tvchannel','cinemas','cookbook','joke','riddle','telephone','video','train','poetry','flight','epg','health','email','bus','story'] = Field(description="领域")
    intent: Literal['OPEN','SEARCH','REPLAY_ALL','NUMBER_QUERY','DIAL','CLOSEPRICE_QUERY','SEND','LAUNCH','PLAY','REPLY','RISERATE_QUERY','DOWNLOAD','QUERY','LOOK_BACK','CREATE','FORWARD','DATE_QUERY','SENDCONTACTS','DEFAULT','TRANSLATION','VIEW','NaN','ROUTE','POSITION'] = Field(description="意图")
    # solt: Literal['code','Src','startDate_dateOrig','film','endLoc_city','artistRole','location_country','location_area','author','startLoc_city','season','dishNamet','media','datetime_date','episode','teleOperator','questionWord','receiver','ingredient','name','startDate_time','startDate_date','location_province','endLoc_poi','artist','dynasty','area','location_poi','relIssue','Dest','content','keyword','target','startLoc_area','tvchannel','type','song','queryField','awayName','headNum','homeName','decade','payment','popularity','tag','startLoc_poi','date','startLoc_province','endLoc_province','location_city','absIssue','utensil','scoreDescr','dishName','endLoc_area','resolution','yesterday','timeDescr','category','subfocus','theatre','datetime_time'] = Field(description="实体")
    # solt: Optional[str]  = Field(description="实体")
    solt: List[str]  = Field(description="实体")

def tool_test(text):
    result = ExtractionAgent(model_name="qwen-plus").call(f"{text}。", TextDomainEntity)
    return result

