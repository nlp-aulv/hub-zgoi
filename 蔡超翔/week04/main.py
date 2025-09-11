# python自带库
import time
import traceback
import re
import torch
from typing import Union

# 第三方库
from fastapi import FastAPI

# 自己写的模块
from data_schema import TextClassifyRequest, TextClassifyResponse
from model.bert import model_for_bert
from logger import logger

# 新增一个专门用于外卖评价分类的请求模型
from pydantic import BaseModel
class FoodReviewRequest(BaseModel):
    review_text: str  # 用户评价文本

class FoodReviewResponse(BaseModel):
    is_positive: bool  # True表示好评，False表示差评
    confidence: float  # 预测置信度
    error_msg: str = ""  # 错误信息


app = FastAPI()


@app.post("/v1/text-cls/bert")
def bert_classify(req: TextClassifyRequest) -> TextClassifyResponse:
    """
    利用BERT进行文本分类

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
    # info 日志
    try:
        logits = model_for_bert(req.request_text)
        response.classify_result = logits
        response.error_msg = "ok"
    except Exception as err:
        # error 日志
        response.classify_result = []
        response.error_msg = traceback.format_exc()

    response.classify_time = round(time.time() - start_time, 3)
    return response


@app.post("/v1/food-review/cls")
def food_review_classify(req: FoodReviewRequest) -> FoodReviewResponse:
    """
    外卖评价分类接口
    输入用户评价文本，返回好评/差评分类结果
    
    参数:
        req: 包含review_text的请求体
        
    返回:
        is_positive: True(好评)/False(差评)
        confidence: 预测置信度(0-1)
    """
    start_time = time.time()
    response = FoodReviewResponse(
        is_positive=False,
        confidence=0.0,
        error_msg=""
    )
    
    try:
        # 1. 预处理输入文本
        processed_text = preprocess_review(req.review_text)
        
        # 2. 使用BERT模型进行预测
        logits = model_for_bert(processed_text)  # 假设返回的是模型原始输出
        
        # 3. 解析预测结果 (假设logits是形状为(1,2)的数组)
        positive_prob = torch.softmax(torch.tensor(logits), dim=1)[0][1].item()
        
        response.is_positive = positive_prob > 0.5
        response.confidence = positive_prob if response.is_positive else 1 - positive_prob
        
    except Exception as err:
        response.error_msg = f"分类失败: {str(err)}"
        logger.error(traceback.format_exc())
    
    logger.info(f"分类耗时: {time.time()-start_time:.3f}s | 文本: {req.review_text[:50]}... | 结果: {response.is_positive}")
    return response

def preprocess_review(text: str) -> str:
    """预处理评价文本"""
    # 1. 去除特殊字符
    text = re.sub(r"[^\w\s]", "", text)
    # 2. 去除多余空格
    text = " ".join(text.split())
    # 3. 其他自定义清洗逻辑...
    return text

import os
# 使用绝对路径（注意去掉末尾斜杠）
BERT_MODEL_PERTRAINED_PATH = "D:/ai/PycharmProjects/nlp20/week04/project1/training_code/food_review_bert/final_model"

# 加载模型和分词器
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PERTRAINED_PATH)
model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PERTRAINED_PATH)

path = "D:/ai/PycharmProjects/nlp20/week04/project1/training_code/food_review_bert/final_model"
print(os.path.exists(path))  # 应输出 True
print(os.listdir(path))      # 应列出模型文件
