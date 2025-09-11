from typing import Union, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertForSequenceClassification, AutoTokenizer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tokenizer = AutoTokenizer.from_pretrained('assets/models/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('assets/models/bert-base-chinese', num_labels=2)
CATEGORY_NAME = [0, 1]

model.load_state_dict(torch.load('assets/weights/bert.pt'))
model.to(device)


class NewsDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(int(self.labels[idx]))
        return item


def model_for_bert(request_text: Union[str, List[str]]) -> Union[str, List[str]]:
    classify_result: Union[str, List[str]] = None

    if isinstance(request_text, str):
        request_text = [request_text]
    elif isinstance(request_text, list):
        pass
    else:
        raise Exception('格式不支持')

    test_encoding = tokenizer(list(request_text), truncation=True, padding=True, max_length=30)
    test_dataset = NewsDataset(test_encoding, [0] * len(request_text))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    model.eval()
    pred = []
    for batch in test_dataloader:
        with torch.no_grad():
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs[1]
        logits = logits.detach().cpu().numpy()
        pred += list(np.argmax(logits, axis=1).flatten())

    classify_result = [CATEGORY_NAME[x] for x in pred]
    return classify_result
