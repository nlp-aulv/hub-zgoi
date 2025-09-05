# python自带库
import time
import traceback
from fastapi import FastAPI
from data_schema import TextClassifyResponse, TextClassifyRequest
from bert import model_for_bert
from logger import logger

app = FastAPI()


@app.post("/v1/bert")
def regex_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用正则表达式进行文本分类

    :param req: 请求体
    """
    start_time = time.time()
    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    logger.info(f"{req.request_id} {req.request_text}")  # 打印请求
    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = "ok"
    except Exception as err:
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
