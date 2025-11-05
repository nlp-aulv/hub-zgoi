import json
import codecs

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch

from datasets import Dataset # 自定义数据集
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

train = json.load(open('../Week07/cmrc2018_public/train.json'))
dev = json.load(open('../Week07/cmrc2018_public/dev.json'))

def prepare_dataset(data):
    paragraphs = []
    questions = []
    answers = []

    for paragraph in data['data']:
        context = paragraph['paragraphs'][0]['context']
        for qa in paragraph['paragraphs'][0]['qas']:
            paragraphs.append(context)
            questions.append(qa['question'])
            answers.append({
                'answer_start': [qa['answers'][0]['answer_start']],
                'text': [qa['answers'][0]['text']]
            })

    return paragraphs, questions, answers

# 准备训练和验证数据
train_paragraphs, train_questions, train_answers = prepare_dataset(train)
val_paragraphs, val_questions, val_answers = prepare_dataset(dev)

# 创建数据集字典
train_dataset_dict = {
    'context': train_paragraphs[:1000],
    'question': train_questions[:1000],
    'answers': [x["text"][0] for x in train_answers[:1000]]
}

val_dataset_dict = {
    'context': val_paragraphs[:100],
    'question': val_questions[:100],
    'answers': [x["text"][0] for x in val_answers[:1000]]
}



# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=384):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n现在进行实体识别任务<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

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


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""

    # 对什么模型，以什么方式进行微调
    config = LoraConfig(
        # 任务类型，自回归语言建模
        task_type=TaskType.CAUSAL_LM,

        # 对什么层的默写模块进行高效微调
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
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
        output_dir="./output_qa_Qwen3/",
        per_device_train_batch_size=6,
        gradient_accumulation_steps=4,
        logging_steps=100,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=50,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "现在进行实体识别任务"},
        {"role": "user", "content": text}
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
            max_new_tokens=512,
            do_sample=False,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip().split("。")[0]


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="预测意图"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")

    global train_dataset_dict
    train_data = pd.DataFrame(train_dataset_dict)
    train_data["instruction"] = train_data["question"] + "\n" +train_data["context"]
    train_data = train_data[["instruction", "answers"]]
    train_data.columns = ["instruction", "output"]
    train_data["input"] = ""
    ds = Dataset.from_pandas(train_data)

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "../models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(ds.to_pandas().iloc[:200])
    eval_ds = Dataset.from_pandas(ds.to_pandas()[-200:])

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

    # 5. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 6. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 7. 创建Trainer并开始训练
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

# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_qa_Qwen3/checkpoint-40/")
    model.cpu()

    # 测试预测
    test_text = "谁开发的ChatGPT？" + "\n" + "ChatGPT 是由美国人工智能研究公司 OpenAI 开发的一款大型语言模型。它基于 GPT（Generative Pre-trained Transformer）架构，通过海量的文本数据进行训练，能够理解和生成人类语言，实现与用户的自然对话。"
    result = predict_intent(model, tokenizer, test_text)
    print(f"{test_text}")
    print(f"\n{result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_single_example()