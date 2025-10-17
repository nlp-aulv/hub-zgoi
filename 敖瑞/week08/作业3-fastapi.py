import time
import traceback
from fastapi import FastAPI
import fastapi_cdn_host

from data_schema import TextIdentificationRequest, TextIdentificationResponse
from work1 import prompt_ner
from work2 import tools_ner

app = FastAPI()
fastapi_cdn_host.patch_docs(app)


@app.post('/v1/test-idf/prompt')
def prompt_identification(req: TextIdentificationRequest) -> TextIdentificationResponse:
    """
    提示词 实现 领域识别、意图识别以及实体识别
    :param req: 请求体
    :return: 识别结果
    """
    start_time = time.time()

    response = TextIdentificationResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        identification_result='',
        identification_time=0,
        error_msg=''
    )

    try:
        response.identification_result = prompt_ner(req.request_text)
        response.error_msg = 'ok'
    except Exception as err:
        response.identification_result = ''
        response.error_msg = traceback.format_exc()

    response.identification_time = round(time.time() - start_time, 3)
    return response


@app.post('/v1/text-idf/tools')
def tools_identification(req: TextIdentificationRequest) -> TextIdentificationResponse:
    """
    tools 实现 领域识别、意图识别以及实体识别
    :param req: 请求体
    :return: 识别结果
    """
    start_time = time.time()

    response = TextIdentificationResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        identification_result='',
        identification_time=0,
        error_msg=''
    )

    try:
        response.identification_result = tools_ner(req.request_text)
        response.error_msg = 'ok'
    except Exception as err:
        response.identification_result = ''
        response.error_msg = traceback.format_exc()

    response.identification_time = round(time.time() - start_time, 3)
    return response
