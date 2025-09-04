import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset


dataset_df = pd.read_csv('../assets/dataset/waimai.csv', sep=',')

labels = list(dataset_df['label'])
texts = list(dataset_df['review'])

x_train, x_test, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, stratify=labels)

tokenizer = BertTokenizer.from_pretrained('../assets/models/bert-base-chinese')
model = BertForSequenceClassification.from_pretrained('../assets/models/bert-base-chinese', num_labels=2)

train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)


train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {'accuracy': (predictions == labels).mean()}


training_args = TrainingArguments(
    output_dir='../assets/weights/bert/',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.001,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()

best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print('The best model is located as: ', best_model_path)
    torch.save(best_model.state_dict(), '../assets/weights/bert.pt')
    print('Best model saved to assets/weights/bert.pt')
else:
    print('Could not find the best model checkpoint.')
