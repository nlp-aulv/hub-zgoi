# python自带库
import time
import traceback
from typing import Union
import uvicorn

# 第三方库
import openai
from fastapi import FastAPI
from data_schema import TextClassifyResponse, TextClassifyRequest
from bert import predict

app = FastAPI(
    title="BERT文本分类API",
    description="基于BERT的中文文本分类服务",
    version="1.0.0"
)


@app.post("/v1/reviews-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )
    # info 日志
    try:
        response.classify_result = predict(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


if __name__ == '__main__':
    # 运行fastapi程序
    uvicorn.run(app="main:app", host="127.0.0.1", port=8000, reload=True)
