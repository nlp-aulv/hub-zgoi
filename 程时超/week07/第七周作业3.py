import json
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
import torch
import os


def load_cmrc2018_data(train_path, dev_path):
    """加载CMRC2018数据集"""
    # 加载训练数据
    with open(train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 加载验证数据
    with open(dev_path, 'r', encoding='utf-8') as f:
        dev_data = json.load(f)

    return train_data, dev_data


def prepare_qa_dataset(data, max_samples=None):
    """准备问答数据集"""
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            if qa['answers']:
                answers.append({
                    'answer_start': [qa['answers'][0]['answer_start']],
                    'text': [qa['answers'][0]['text']]
                })
            else:
                answers.append({
                    'answer_start': [],
                    'text': []
                })

    if max_samples:
        paragraphs = paragraphs[:max_samples]
        questions = questions[:max_samples]
        answers = answers[:max_samples]

    return paragraphs, questions, answers


def convert_to_instruction_format(context, question, answer):
    """将QA任务转换为指令格式"""
    if answer['text']:
        answer_text = answer['text'][0]
        output = f"答案：{answer_text}"
    else:
        output = "根据上下文无法找到答案。"

    return {
        "instruction": f"阅读以下文本并回答问题：\n文本：{context}",
        "input": f"问题：{question}",
        "output": output
    }


def prepare_instruction_dataset(train_data, dev_data, max_train_samples=800, max_dev_samples=200):
    """准备指令微调数据集"""
    # 准备原始数据
    train_paragraphs, train_questions, train_answers = prepare_qa_dataset(
        train_data, max_samples=max_train_samples
    )
    dev_paragraphs, dev_questions, dev_answers = prepare_qa_dataset(
        dev_data, max_samples=max_dev_samples
    )

    # 转换为指令格式
    train_instructions = []
    for context, question, answer in zip(train_paragraphs, train_questions, train_answers):
        train_instructions.append(convert_to_instruction_format(context, question, answer))

    dev_instructions = []
    for context, question, answer in zip(dev_paragraphs, dev_questions, dev_answers):
        dev_instructions.append(convert_to_instruction_format(context, question, answer))

    # 转换为DataFrame
    train_df = pd.DataFrame(train_instructions)
    dev_df = pd.DataFrame(dev_instructions)

    return train_df, dev_df


def initialize_model_and_tokenizer(model_path):
    """初始化模型和tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    return tokenizer, model


def process_func(example, tokenizer, max_length=512):
    """处理单个样本的函数"""
    # 构建完整的对话文本
    system_msg = "你是一个知识问答助手，需要根据给定的文本内容回答问题。"
    instruction_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"

    # Tokenize指令部分
    instruction_tokens = tokenizer(instruction_text, add_special_tokens=False)

    # Tokenize响应部分
    response_tokens = tokenizer(example['output'], add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction_tokens["input_ids"] + response_tokens["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction_tokens["attention_mask"] + response_tokens["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction_tokens["input_ids"]) + response_tokens["input_ids"] + [tokenizer.pad_token_id]

    # 截断超过最大长度的序列
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def setup_lora(model):
    """设置LoRA配置"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def predict_answer(model, tokenizer, context, question, device='cuda'):
    """预测答案"""
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，需要根据给定的文本内容回答问题。"},
        {"role": "user", "content": f"阅读以下文本并回答问题：\n文本：{context}\n问题：{question}"}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


def main():
    """主函数"""
    # 1. 加载数据
    print("加载数据集...")
    train_data, dev_data = load_cmrc2018_data(
        './cmrc2018_public/train.json',
        './cmrc2018_public/dev.json'
    )

    # 2. 准备指令数据集
    print("准备指令微调数据集...")
    train_df, dev_df = prepare_instruction_dataset(
        train_data, dev_data,
        max_train_samples=800,
        max_dev_samples=200
    )

    # 保存数据集用于检查
    train_df.to_csv('qa_train_data.csv', index=False, encoding='utf-8')
    dev_df.to_csv('qa_dev_data.csv', index=False, encoding='utf-8')

    print(f"训练集大小: {len(train_df)}")
    print(f"验证集大小: {len(dev_df)}")

    # 3. 初始化模型
    print("初始化Qwen模型...")
    model_path = "../models/Qwen/Qwen3-0.6B"  # 根据实际路径调整
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 4. 处理数据
    print("处理训练数据...")
    train_dataset = Dataset.from_pandas(train_df)
    dev_dataset = Dataset.from_pandas(dev_df)

    # 创建处理函数
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    tokenized_train = train_dataset.map(
        process_func_with_tokenizer,
        remove_columns=train_dataset.column_names,
        batched=False
    )
    tokenized_dev = dev_dataset.map(
        process_func_with_tokenizer,
        remove_columns=dev_dataset.column_names,
        batched=False
    )

    # 5. 设置LoRA
    print("设置LoRA配置...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 6. 设置训练参数
    training_args = TrainingArguments(
        output_dir="./qwen-qa-model",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=50,
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="none",
        warmup_steps=100,
        logging_dir="./logs",
    )

    # 7. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    # 8. 开始训练
    print("开始训练Qwen问答模型...")
    trainer.train()

    # 9. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-qa-model")

    return model, tokenizer, train_data, dev_data

def load_trained_model(model_path, base_model_path="../models/Qwen/Qwen3-0.6B"):
    """加载训练好的模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 设置padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    # 加载LoRA适配器
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return tokenizer, model

def predict_answer(model, tokenizer, context, question, device='cuda'):
    """预测答案"""
    messages = [
        {"role": "system", "content": "你是一个知识问答助手，需要根据给定的文本内容回答问题。"},
        {"role": "user", "content": f"阅读以下文本并回答问题：\n文本：{context}\n问题：{question}"}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


if __name__ == "__main__":
    if os.path.exists('./qwen-qa-model'):
        pass
    else:
        # 训练模型
        model, tokenizer, train_data, dev_data = main()

    # 加载训练好的模型
    model_path = "./qwen-qa-model"  # 你保存的模型路径
    tokenizer, model = load_trained_model(model_path)

    # 测试预测
    context = "清华大学位于北京市海淀区，是中国著名的综合性大学。"
    question = "清华大学位于哪里？"

    answer = predict_answer(model, tokenizer, context, question)
    print(f"问题: {question}")
    print(f"答案: {answer}")
