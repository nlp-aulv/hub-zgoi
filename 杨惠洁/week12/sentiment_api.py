from http.client import HTTPException
from typing import Dict

import openai
import uvicorn
from fastapi import FastAPI

TOKEN = "738b541a5f7a"
app = FastAPI(title="情感分析接口", description="基于用户语言的情感分析", version="0.1.0")

client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-cc43ca2821f64bfa9c6e20bf0889d92c",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 健康检查端点
@app.get("/")
async def root():
    return {"status": "running", "message": "情感分析服务已启动"}


@app.get("/sentiment")
async def sentiment(text: str, key: str = TOKEN):
    """
    情感分析接口
    """
    if key != TOKEN:
        raise HTTPException(status_code=401, detail="无效的 token")
    try:
        response = client.chat.completions.create(
            model="qwen-flash",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个情感分析助手，请用一句话说出用户的话是什么样的情感状态,用消极，积极，中性来回答"
                },
                {
                    "role": "user",
                    "content": f"分析这句话的情感：{text}"
                }
            ]
        )

        result = response.choices[0].message.content


        return {
            "sentiment_result": result,
            "status": "success"
        }

    except Exception as e:
        return {
            "sentiment_result": "无法分析",
            "status": "error",
            "message": str(e)
        }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)