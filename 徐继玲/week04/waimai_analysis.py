import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import load_dataset, DatasetDict
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List
import os
from sklearn.metrics import accuracy_score
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextInput(BaseModel):
    text: str


class BatchTextInput(BaseModel):
    texts: List[str]


def load_and_prepare_data():
    """
    加载和预处理外卖评价数据集
    """
    try:
        # 从Hugging Face加载数据集[1,2](@ref)
        dataset = load_dataset('XiangPan/waimai_10k')
        logger.info("数据集加载成功")

        # 检查数据集结构并重命名列（如果需要）
        if 'review' in dataset['train'].column_names and 'label' in dataset['train'].column_names:
            # 重命名列以符合标准[2](@ref)
            dataset = dataset.rename_columns({'review': 'text', 'label': 'labels'})

        # 划分训练集和测试集（80%训练，20%测试）
        train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)

        # 创建DatasetDict[6](@ref)
        dataset_dict = DatasetDict({
            'train': train_test_split['train'],
            'test': train_test_split['test']
        })

        return dataset_dict

    except Exception as e:
        logger.error(f"加载数据时出错: {str(e)}")
        raise


def fine_tune_bert():
    """
    微调BERT模型进行情感分析
    """
    # 加载数据集
    dataset = load_and_prepare_data()

    # 加载预训练模型和分词器[1,3](@ref)
    model_name = "bert-base-chinese"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # 数据预处理函数
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,  # 使用DataCollator进行动态填充
            truncation=True,
            max_length=128
        )

    # 应用分词函数
    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    # 设置数据格式
    tokenized_datasets.set_format("torch")

    # 创建数据收集器用于动态填充
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 加载模型[1](@ref)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2
    )

    # 定义训练参数[1](@ref)
    training_args = TrainingArguments(
        output_dir="./bert_waimai_results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
    )

    # 定义计算指标的函数
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": accuracy_score(labels, predictions)}

    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 开始训练
    logger.info("开始训练模型...")
    trainer.train()

    # 保存模型和分词器[2](@ref)
    model.save_pretrained("./my_bert_model")
    tokenizer.save_pretrained("./my_bert_model")
    logger.info("模型保存完成")

    return model, tokenizer


# 初始化全局变量
model = None
tokenizer = None
app = FastAPI(title="外卖评价情感分析API", description="基于BERT的美团外卖评价情感分析服务")


def load_model():
    """
    加载已训练的模型
    """
    global model, tokenizer
    try:
        model = AutoModelForSequenceClassification.from_pretrained("./my_bert_model")
        tokenizer = AutoTokenizer.from_pretrained("./my_bert_model")
        logger.info("模型加载成功")
    except:
        logger.warning("未找到已训练的模型，开始微调新模型...")
        model, tokenizer = fine_tune_bert()


@app.on_event("startup")
async def startup_event():
    """
    应用启动时加载模型
    """
    load_model()


@app.post("/predict")
async def predict_sentiment(input_data: TextInput):
    """
    单条文本情感预测端点
    """
    try:
        # 对输入文本进行编码
        inputs = tokenizer(
            input_data.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )

        # 模型推理
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # 获取预测结果
        predicted_class_id = predictions.argmax().item()
        confidence = predictions[0][predicted_class_id].item()

        # 将数字标签转换为有意义的名称
        label_name = "差评" if predicted_class_id == 0 else "好评"

        return {
            "text": input_data.text,
            "predicted_label": label_name,
            "confidence": confidence,
            "class_id": predicted_class_id
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_batch")
async def predict_batch_sentiment(batch_data: BatchTextInput):
    """
    批量文本情感预测端点
    """
    try:
        results = []
        for text in batch_data.texts:
            # 对每条文本进行编码和预测
            inputs = tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            )

            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

            predicted_class_id = predictions.argmax().item()
            confidence = predictions[0][predicted_class_id].item()
            label_name = "差评" if predicted_class_id == 0 else "好评"

            results.append({
                "text": text,
                "predicted_label": label_name,
                "confidence": confidence,
                "class_id": predicted_class_id
            })

        return {"results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy", "model_loaded": model is not None}


@app.get("/")
async def root():
    """
    根端点
    """
    return {
        "message": "外卖评价情感分析服务",
        "endpoints": {
            "/predict": "POST - 单条文本情感分析",
            "/predict_batch": "POST - 批量文本情感分析",
            "/health": "GET - 健康检查",
            "/docs": "GET - API文档"
        }
    }


if __name__ == "__main__":
    # 启动FastAPI应用
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )