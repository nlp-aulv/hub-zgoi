import pandas as pd
import torch
import re
from sklearn.model_selection import train_test_split
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from datasets import Dataset, ClassLabel
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(file_path):
    """加载和预处理外卖评价数据"""
    # 读取CSV文件，明确指定header=0表示第一行是列名
    df = pd.read_csv(file_path, header=0, names=['label', 'review'], sep=',')
    
    logger.info(f"原始数据样例:\n{df.head()}")
    
    # 数据清洗函数
    def clean_review(text):
        if not isinstance(text, str):  # 处理可能的非字符串数据
            return ""
        text = re.sub(r'[^\w\s，。！？]', '', text)  # 保留中文标点
        text = re.sub(r'\s+', '', text)  # 去除所有空白字符
        return text.strip()
    
    # 应用清洗
    df['review'] = df['review'].apply(clean_review)
    
    # 过滤空数据
    df = df[df['review'].str.len() > 0]
    
    # 确保标签是0或1 (0=差评, 1=好评)
    try:
        df['label'] = pd.to_numeric(df['label']).astype(int)
        df = df[df['label'].isin([0, 1])]
    except Exception as e:
        logger.error(f"标签转换错误: {e}")
        raise ValueError("标签列必须只包含0或1的整数值")
    
    logger.info(f"\n清洗后数据分布:\n{df['label'].value_counts()}")
    logger.info(f"\n示例数据:\n{df.sample(3)}")
    
    return df

def tokenize_data(texts, tokenizer, max_length=64):
    """针对中文评价优化的tokenize处理"""
    return tokenizer(
        texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True  # 添加[CLS]和[SEP]
    )

def compute_metrics(eval_pred):
    """计算评估指标"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='macro')
    }

def main():
    # 1. 加载数据
    data_file = "./waimai_10k.csv"  # 使用您提供的文件名
    df = load_and_preprocess_data(data_file)
    
    # 2. 分割数据集 (80%训练，20%测试)
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df['label'],
        random_state=42
    )
    
    # 3. 初始化中文BERT模型
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-chinese",
        num_labels=2,
        id2label={0: "差评", 1: "好评"},
        label2id={"差评": 0, "好评": 1}
    )
    
    # 4. 准备数据集
    train_encodings = tokenize_data(train_df['review'].tolist(), tokenizer)
    test_encodings = tokenize_data(test_df['review'].tolist(), tokenizer)
    
    # 转换为PyTorch Dataset
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': torch.tensor(train_df['label'].values)
    })
    
    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': torch.tensor(test_df['label'].values)
    })

    # 5. 训练配置（针对中文短文本优化）
    training_args = TrainingArguments(
        output_dir='./food_review_bert',
        num_train_epochs=8,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=100,
        evaluation_strategy="steps",  # 改为steps
        eval_steps=200,               # 每200步评估一次
        save_strategy="steps",        # 保存策略与评估策略一致
        save_steps=200,               # 保存步数与评估步数相同
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=True,
        report_to="none",
        logging_dir='./logs',
        save_total_limit=2            # 最多保存2个检查点
    )

    # 6. 自定义评估指标
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            'accuracy': accuracy_score(labels, preds),
            'f1': f1_score(labels, preds),
            'precision': precision_score(labels, preds),
            'recall': recall_score(labels, preds)
        }
    
    # 7. 开始训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    
    logger.info("开始训练中文外卖评价分类模型...")
    train_result = trainer.train()
    
    # 8. 保存最终模型
    trainer.save_model("./food_review_bert/final_model")
    tokenizer.save_pretrained("./food_review_bert/final_model")
    
    # 评估测试集
    eval_results = trainer.evaluate(test_dataset)
    logger.info(f"最终测试集性能:\n{eval_results}")

if __name__ == "__main__":
    main()
