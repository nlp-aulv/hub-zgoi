import codecs
import shutil

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
import torch


# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


# 数据加载和预处理
def load_and_preprocess_data(tokenizer):
    # 加载训练数据
    train_lines = codecs.open('./msra/train/sentences.txt', encoding='utf-8').readlines()[:1000]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt', encoding='utf-8').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]
    train_tags = [[label2id[x] for x in tag] for tag in train_tags]

    # 检查标签范围
    for i, tag_list in enumerate(train_tags):
        for tag in tag_list:
            if tag >= len(tag_type):
                raise ValueError(f"标签值 {tag} 超出范围，最大允许值为 {len(tag_type) - 1}")
    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt', encoding='utf-8').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt', encoding='utf-8').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]
    val_tags = [[label2id[x] for x in tag] for tag in val_tags]

    train_dataset = prepare_dataset(train_lines, train_tags, tokenizer)
    eval_dataset = prepare_dataset(val_lines, val_tags, tokenizer)
    # 检查标签范围
    for i, tag_list in enumerate(val_tags):
        for tag in tag_list:
            if tag >= len(tag_type):
                raise ValueError(f"标签值 {tag} 超出范围，最大允许值为 {len(tag_type) - 1}")

    return train_dataset, eval_dataset


def prepare_dataset(texts, tags, tokenizer):
    # 将文本拆分为字符列表
    tokens = [list(text) for text in texts]

    # 创建数据集
    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })

    # 对数据集进行tokenize
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        padding=False,
        max_length=128,
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples["labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            # 特殊token设置为-100，在计算损失时会被忽略
            if word_idx is None:
                label_ids.append(-100)
            # 当前单词与之前单词相同（子词）
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # 当前单词是之前单词的一部分
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(tag_type),  # 确保与 tag_type 长度一致
        device_map="auto",
        torch_dtype=torch.float16
    )

    return tokenizer, model


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,  # 使用Token分类
        target_modules=["q_proj", "v_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_Qwen1.5",
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_strategy='no',
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=False,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_entities(model, tokenizer, text, device='cuda'):
    """预测实体"""
    # 将文本拆分为字符列表
    tokens = list(text)

    # Tokenize 输入
    model_inputs = tokenizer(
        tokens,
        truncation=True,
        max_length=128,
        is_split_into_words=True,
        return_tensors="pt"
    ).to(device)

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits

    # 获取预测标签
    predictions = logits.argmax(dim=-1)[0].tolist()

    # 将预测标签映射为实体标签
    entities = []
    current_entity = None
    for token, label_id in zip(tokens, predictions):
        label = id2label[label_id]
        if label != "O":
            if current_entity is None:
                current_entity = {"text": token, "type": label[2:], "start": len(entities)}
            else:
                current_entity["text"] += token
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


# 主函数
def main():
    """主执行函数"""
    # 清空输出目录
    if os.path.exists("./output_Qwen1.5"):
        shutil.rmtree("./output_Qwen1.5")
    os.makedirs("./output_Qwen1.5", exist_ok=True)

    # 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载数据
    print("加载数据...")
    train_tokenized, eval_tokenized = load_and_preprocess_data(tokenizer)

    # 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 保存模型和 LoRA 权重
    print("保存模型和 LoRA 权重...")
    model.save_pretrained("./output_Qwen1.5/")  # 保存 LoRA 权重
    tokenizer.save_pretrained("./output_Qwen1.5/")  # 保存分词器


# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen1.5/")
    model.to("cuda")

    # 测试预测
    test_text = "江 主 席 来 到 哈 佛 大 学 时 ， 受 到 哈 佛 大 学 校 长 陆 登 庭 及 哈 佛 各 学 院 院 长 的 热 烈 欢 迎 。"
    result = predict_entities(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"实体抽取: {result}")


if __name__ == "__main__":
    # 执行主函数
    # main()

    # 单独测试
    test_single_example()
