import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model
import torch
from tqdm import tqdm
import codecs
import json

# 定义实体类型标签
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


def load_msra_data(file_path, tags_path, max_samples=1000):
    """加载MSRA数据集"""
    # 加载文本
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.replace(' ', '').strip() for line in f.readlines()[:max_samples]]

    # 加载标签
    with open(tags_path, 'r', encoding='utf-8') as f:
        tags = [line.strip().split(' ') for line in f.readlines()[:max_samples]]

    return sentences, tags


def convert_to_instruction_format(sentence, tags):
    """将实体识别任务转换为指令格式"""
    # 提取实体
    entities = []
    current_entity = ""
    current_type = ""

    for char, tag in zip(sentence, tags):
        if tag.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = tag[2:]
        elif tag.startswith('I-') and current_entity and current_type == tag[2:]:
            current_entity += char
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""

    if current_entity:
        entities.append((current_entity, current_type))

    # 构建指令
    if entities:
        entities_str = "，".join([f"'{entity}'({etype})" for entity, etype in entities])
        output = f"文本中的实体有：{entities_str}"
    else:
        output = "文本中没有识别到实体"

    return {
        "instruction": f"请从以下文本中识别出人名(PER)、地名(LOC)和组织名(ORG)实体：{sentence}",
        "input": "",
        "output": output
    }


def prepare_ner_dataset():
    """准备NER数据集"""
    # 加载训练数据
    train_sentences, train_tags = load_msra_data(
        './msra/train/sentences.txt',
        './msra/train/tags.txt',
        max_samples=800
    )

    # 加载验证数据
    val_sentences, val_tags = load_msra_data(
        './msra/val/sentences.txt',
        './msra/val/tags.txt',
        max_samples=200
    )

    # 转换为指令格式
    train_data = []
    for sentence, tags in zip(train_sentences, train_tags):
        train_data.append(convert_to_instruction_format(sentence, tags))

    val_data = []
    for sentence, tags in zip(val_sentences, val_tags):
        val_data.append(convert_to_instruction_format(sentence, tags))

    # 转换为DataFrame
    train_df = pd.DataFrame(train_data)
    val_df = pd.DataFrame(val_data)

    return train_df, val_df


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
    """处理单个样本"""
    # 构建指令文本
    instruction_text = f"<|im_start|>system\n你是一个实体识别专家，需要从文本中识别出人名、地名和组织名实体。<|im_end|>\n<|im_start|>user\n{example['instruction']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(example['output'], add_special_tokens=False)

    # 组合输入
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]

    # 截断
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


def predict_entities(model, tokenizer, text, device='cuda'):
    """预测实体"""
    messages = [
        {"role": "system", "content": "你是一个实体识别专家，需要从文本中识别出人名、地名和组织名实体。"},
        {"role": "user", "content": f"请从以下文本中识别出人名(PER)、地名(LOC)和组织名(ORG)实体：{text}"}
    ]

    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


def main():
    """主函数"""
    # 1. 准备数据
    print("准备数据集...")
    train_df, val_df = prepare_ner_dataset()

    # 保存数据集用于检查
    train_df.to_csv('ner_train_data.csv', index=False, encoding='utf-8')
    val_df.to_csv('ner_val_data.csv', index=False, encoding='utf-8')

    # 2. 初始化模型
    print("初始化模型...")
    model_path = "../models/Qwen/Qwen3-0.6B"  # 根据实际路径调整
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理数据...")
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)

    tokenized_train = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    tokenized_val = val_dataset.map(process_func_with_tokenizer, remove_columns=val_dataset.column_names)

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 训练参数
    training_args = TrainingArguments(
        output_dir="./qwen-ner-model",
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
    )

    # 6. 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    # 7. 开始训练
    print("开始训练...")
    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./qwen-ner-model")

    return model, tokenizer


def test_model(model, tokenizer):
    """测试训练好的模型"""
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
    ]

    print("测试模型预测效果：")
    for sentence in test_sentences:
        try:
            result = predict_entities(model, tokenizer, sentence)
            print(f"文本: {sentence}")
            print(f"实体识别结果: {result}")
            print("-" * 50)
        except Exception as e:
            print(f"处理句子时出错: {sentence}")
            print(f"错误: {e}")
            print("-" * 50)


if __name__ == "__main__":
    # 训练模型
    model, tokenizer = main()

    # 测试模型
    test_model(model, tokenizer)
