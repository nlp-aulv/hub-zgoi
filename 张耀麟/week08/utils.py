import regex
from pydantic import BaseModel, Field
from typing import Dict, Tuple
from typing_extensions import Literal

import openai
import json
import re

SYSTEM_PROMPT = """
                    请你帮我从给定句子(text)中提取领域(domain),意图(intent),命名实体(slots),
                    需要提取的内容如下:
                    领域(domain):code, Src, startDate_dateOrig, film, endLoc_city, artistRole, location_country, location_area, author, startLoc_city, season, dishNamet, media, datetime_date, episode, teleOperator, questionWord, receiver, ingredient, name, startDate_time, startDate_date, location_province, endLoc_poi, artist, dynasty, area, location_poi, relIssue, Dest, content, keyword, target, startLoc_area, tvchannel, type, song, queryField, awayName, headNum, homeName, decade, payment, popularity, tag, startLoc_poi, date, startLoc_province, endLoc_province, location_city, absIssue, utensil, scoreDescr, dishName, endLoc_area, resolution, yesterday, timeDescr, category, subfocus, theatre, datetime_time.
                    意图(intent):music, app, radio, lottery, stock, novel, weather, match, map, website, news, message, contacts, translation, tvchannel, cinemas, cookbook, joke, riddle, telephone, video, train, poetry, flight, epg, health, email, bus, story.
                    命名实体(slots):SEARCH, REPLAY_ALL, NUMBER_QUERY, DIAL, CLOSEPRICE_QUERY, SEND, LAUNCH, PLAY, REPLY, RISERATE_QUERY, DOWNLOAD, QUERY, LOOK_BACK, CREATE, FORWARD, DATE_QUERY, SENDCONTACTS, DEFAULT, TRANSLATION, VIEW, NaN, ROUTE, POSITION.
                    
                    请注意以JSON format格式返回,可以通过json.loads()解析，包含如下内容:
                    [{
                        "text": 给定的句子,
                        "domain": 提取的领域,
                        "intent": 提取的意图,
                        "slots": {
                          "提取的实体类别": 实体
                        }
                    }]
                    其中命名实体(slots)可以包含多个实体,领域(domain)与意图(intent)只能是唯一一个。
                    你可以参考如下例子:
                    [{
                        "text": "请帮我打开uc",
                        "domain": "app",
                        "intent": "LAUNCH",
                        "slots": {
                          "name": "uc"
                        }
                      },
                      {
                        "text": "打开汽车之家",
                        "domain": "app",
                        "intent": "LAUNCH",
                        "slots": {
                          "name": "汽车之家"
                        }
                      },
                      {
                        "text": "帮我打开人人",
                        "domain": "app",
                        "intent": "LAUNCH",
                        "slots": {
                          "name": "人人"
                        }
                      }
                    }]
                """

DOMAIN = ("code", "Src", "startDate_dateOrig", "film", "endLoc_city", "artistRole", "location_country",
          "location_area", "author", "startLoc_city", "season", "dishNamet", "media", "datetime_date",
          "episode", "teleOperator", "questionWord", "receiver", "ingredient", "name", "startDate_time",
          "startDate_date", "location_province", "endLoc_poi", "artist", "dynasty", "area", "location_poi",
          "relIssue", "Dest", "content", "keyword", "target", "startLoc_area", "tvchannel", "type", "song",
          "queryField", "awayName", "headNum", "homeName", "decade", "payment", "popularity", "tag",
          "startLoc_poi", "date", "startLoc_province", "endLoc_province", "location_city", "absIssue",
          "utensil", "scoreDescr", "dishName", "endLoc_area", "resolution", "yesterday", "timeDescr",
          "category", "subfocus", "theatre", "datetime_time")

INTENT = ("music", "app", "radio", "lottery", "stock", "novel", "weather", "match", "map", "website",
          "news", "message", "contacts", "translation", "tvchannel", "cinemas", "cookbook", "joke",
          "riddle", "telephone", "video", "train", "poetry", "flight", "epg", "health", "email", "bus",
          "story")

SLOTS = ("SEARCH", "REPLAY_ALL", "NUMBER_QUERY", "DIAL", "CLOSEPRICE_QUERY", "SEND", "LAUNCH", "PLAY",
         "REPLY", "RISERATE_QUERY", "DOWNLOAD", "QUERY", "LOOK_BACK", "CREATE", "FORWARD", "DATE_QUERY",
         "SENDCONTACTS", "DEFAULT", "TRANSLATION", "VIEW", "NaN", "ROUTE", "POSITION")


class Tools(BaseModel):
    """请你根据提供的句子提取领域(domain),意图(intent),命名实体(slots)"""
    domain: Literal[DOMAIN] = Field(description="领域")
    intent: Literal[INTENT] = Field(description="意图")
    slots: Dict[Literal[SLOTS], str] = Field(description="实体")


class LLM:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = openai.OpenAI(
            api_key="sk-0d5c48a655774117bf8e4dfea9eb9f7f",
            base_url="https://api.deepseek.com",
        )

    def prompt_call(self, user_prompt: str):
        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )

        content = response.choices[0].message.content
        # 大模型会返回如下格式```json [{}]```，通过正则表达式进行匹配
        pattern = re.compile(r'\{[\s\S]*}')
        match = pattern.search(content)
        if not match:
            print("re match error")
            return None
        return json.loads(match.group())

    def call(self, user_prompt):
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
                    "name": Tools.model_json_schema()['title'],
                    "description": Tools.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": Tools.model_json_schema()['properties'],
                        "required": Tools.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return json.loads(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


if __name__ == "__main__":
    llm = LLM("deepseek-chat")
    rep_prompt = llm.prompt_call("恩，红烧土豆怎么做我想学？")
    print(rep_prompt)

    rep = llm.call("恩，红烧土豆怎么做我想学？")
    print(rep)
