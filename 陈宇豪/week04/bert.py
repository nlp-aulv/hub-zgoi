from typing import Union, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModelForMaskedLM, BertForSequenceClassification
from constants import *
from train_bert import DATAGET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 导入训练模型和分词器
model = BertForSequenceClassification.from_pretrained(SAVE_PATH)
tokenizer = AutoTokenizer.from_pretrained(SAVE_PATH)

model = model.to(device)


# 预测方法
def predict(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    result: Union[str, List[str]] = None
    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception("格式不支持")
    predict_encoding = tokenizer(list(request_text), truncation=True, padding=True, return_tensors="pt", max_length=64)
    predict_data = DATAGET(predict_encoding, pd.Series(np.zeros(len(request_text))))
    predict_data_loader = DataLoader(dataset=predict_data, batch_size=32, shuffle=False)
    model.eval()
    pred = []

    for batch in predict_data_loader:
        with torch.no_grad():
            inputs = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model(inputs, attention_mask=attention_mask)
        logits = outputs.logits
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())
    result = [CATEGORY_DIC[x] for x in pred]
    return result


if __name__ == '__main__':
    print(predict("不好吃"))
