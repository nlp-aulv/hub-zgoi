from fastapi import FastAPI
import numpy as np
from model.bert import model_for_bert
from data_schema import TextClassifyRequest
from data_schema import TextClassifyResponse
import traceback
import time
from logger import logger
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/bert-takeout-evaluation-results")
async def get_bert_main_train_data(req: TextClassifyRequest) -> TextClassifyResponse:
    """
       利用Bert进行文本分类
       :param req: 请求体
       """
    start_time = time.time()
    # logger.info("请求参数：",req)
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    logger.info("执行-start:")
    # info 日志
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        logger.info("执行-end:")
        response.classify_result = ""
        response.error_msg = traceback.format_exc()


    response.classify_time = round(time.time() - start_time, 3)
    return response