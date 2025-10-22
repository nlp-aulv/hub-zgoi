import re

import openai
import json

def prompt_test(text):
    client = openai.OpenAI(
        api_key="sk-cb217fdebe8c46dbb7e5e6aa76f9da89",  # https://bailian.console.aliyun.com/?tab=model#/api-key
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": f"""你是一个文本助手，请你从如下的文本进行意图识别、领域识别、实体识别
            意图有:OPEN/SEARCH/REPLAY_ALL/NUMBER_QUERY/DIAL/CLOSEPRICE_QUERY/SEND/LAUNCH/PLAY/REPLY/RISERATE_QUERY/DOWNLOAD/QUERY/LOOK_BACK/CREATE/FORWARD/DATE_QUERY/SENDCONTACTS/DEFAULT/TRANSLATION/VIEW/NaN/ROUTE/POSITION
            领域有:music/app/radio/lottery/stock/novel/weather/match/map/website/news/message/contacts/translation/tvchannel/cinemas/cookbook/joke/riddle/telephone/video/train/poetry/flight/epg/health/email/bus/story
            实体有:code/Src/startDate_dateOrig/film/endLoc_city/artistRole/location_country/location_area/author/startLoc_city/season/dishNamet/media/datetime_date/episode/teleOperator/questionWord/receiver/ingredient/name/startDate_time/startDate_date/location_province/endLoc_poi/artist/dynasty/area/location_poi/relIssue/Dest/content/keyword/target/startLoc_area/tvchannel/type/song/queryField/awayName/headNum/homeName/decade/payment/popularity/tag/startLoc_poi/date/startLoc_province/endLoc_province/location_city/absIssue/utensil/scoreDescr/dishName/endLoc_area/resolution/yesterday/timeDescr/category/subfocus/theatre/datetime_time
            domain为意图,intent为领域,slots为实体,输出返回如下的json格式
    {{
        "text": "{text}",
        "domain": "",
        "intent": "",
        "slots": {{}}
      }}
            """
             },
            {"role": "user", "content": f"{text}"},
        ],
    )

    # return completion.choices[0].message.content

    raw_output = completion.choices[0].message.content.strip()

    # 移除 markdown 代码块标记
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_output.strip(), flags=re.MULTILINE).strip()

    # 尝试解析 JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"raw_output": raw_output, "error": "Model did not return valid JSON"}

