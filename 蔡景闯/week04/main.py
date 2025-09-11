import traceback
from typing import Dict, List, Any, Union, Optional
from fastapi import FastAPI
from pydantic import BaseModel, Field
import time
from transformers import BertTokenizer
from transformers import BertForSequenceClassification
import torch
from fastapi.responses import HTMLResponse

# 模型和分词器加载
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('./models/google-bert/bert-base-chinese',
                                                      num_labels=2)
model.load_state_dict(torch.load('./bert_model.pth'))
tokenizer = BertTokenizer.from_pretrained('./models/google-bert/bert-base-chinese')
class Response(BaseModel):
    request_id: Optional[str] = Field(..., description="请求id")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")
    classify_result: Union[str, List[str]] = Field(..., description="分类结果")
    classify_time: float = Field(..., description="分类耗时")
    error_msg: str = Field(..., description="异常信息")


class Request(BaseModel):
    request_id: Optional[str] = Field(..., description="请求id, 方便调试")
    request_text: Union[str, List[str]] = Field(..., description="请求文本、字符串或列表")


def bert_predict(text: Union[str, List[str]]) -> Union[int, List[int]]:
    encoding = tokenizer(text, truncation=True, padding=True, max_length=128)
    # 将编码数据转换为PyTorch张量
    input_ids = torch.tensor(encoding['input_ids']).unsqueeze(0).to(device)
    attention_mask = torch.tensor(encoding['attention_mask']).unsqueeze(0).to(device)
    # 将模型加载到GPU或CPU
    model.to(device)
    model.eval()
    # 执行前向传播，得到logits
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
    # 预测结果
    prediction = torch.argmax(logits, dim=1).item()
    return prediction


app = FastAPI()

@app.get("/", response_class=HTMLResponse)
def root():
    return """
        <html>
        <head>
            <title>BERT文本分类</title>
        </head>
        <body>
            <h1>BERT文本分类</h1>
        </body>
        </html>
        """

@app.post("/v1/ReviewClassify/bert")
def review_classify_bert(req: Request) -> Response:
    start_time = time.time()
    # 创建响应对象
    response = Response(
        request_id=req.request_id,
        request_text=req.request_text,
        classify_result="",
        classify_time=0,
        error_msg=""
    )

    try:
        response.classify_result = "好评" if bert_predict(req.request_text) == 1 else "差评"
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = ""
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
    # 压力测试
    # ab -n 100 -c 10 -p C:/ApacheBench/request.json -T "application/json" -H "accept: application/json" "http://127.0.0.1:8000/v1/ReviewClassify/bert"
