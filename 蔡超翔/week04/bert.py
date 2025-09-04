from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 使用绝对路径指向训练好的模型
BERT_MODEL_PATH = "D:/ai/PycharmProjects/nlp20/week04/project1/training_code/food_review_bert/final_model"

# 验证模型路径是否存在
if not os.path.exists(BERT_MODEL_PATH):
    raise FileNotFoundError(f"BERT模型路径不存在: {BERT_MODEL_PATH}")

# 加载tokenizer和模型
try:
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    print("✅ BERT模型和tokenizer加载成功")
except Exception as e:
    raise RuntimeError(f"加载BERT模型失败: {str(e)}")


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # 读取单个样本
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item

    def __len__(self):
        return len(self.labels)


def model_for_bert(text: str):
    """
    使用BERT模型进行文本分类
    :param text: 输入文本
    :return: 分类结果的logits
    """
    try:
        # 文本预处理和tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding="max_length"
        )
        
        # 模型预测
        with torch.no_grad():
            outputs = model(**inputs)
        
        # 将 NumPy 数组转换为 Python 列表
        return outputs.logits.numpy().tolist()  # 关键修改：添加 .tolist()
    
    except Exception as e:
        raise RuntimeError(f"文本分类失败: {str(e)}")

# 测试函数
if __name__ == "__main__":
    test_text = "这家餐厅的菜品很好吃"
    print(f"测试文本: {test_text}")
    logits = model_for_bert(test_text)
    print(f"预测结果: {logits}")
