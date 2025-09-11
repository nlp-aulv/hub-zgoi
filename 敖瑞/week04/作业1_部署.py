import time
import traceback
from fastapi import FastAPI
import fastapi_cdn_host

from model.bert import model_for_bert
from data_schema import TextClassifyResponse, TextClassifyRequest


app = FastAPI()
fastapi_cdn_host.patch_docs(app)


@app.post('/v1/text-cls/bert')
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    start_time = time.time()

    response = TextClassifyResponse(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result='',
        classify_time=0,
        error_msg=''
    )

    try:
        response.classify_result = model_for_bert(req.request_text)
        response.error_msg = 'ok'
    except Exception as err:
        response.classify_result = ''
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response
