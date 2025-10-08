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
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import codecs
import torch
import json

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}
SYSTEM_PROMPT = """
                    你是一个文本实体识别领域的专家，你需要从给定的句子中提取:
                    ORG,
                    PER,
                    LOC, 
                    以如下格式输出: LOC:南京;LOC:北京
                    注意: 
                    1. 多个命名实体使用;分隔
                    2. 不要有其他多余信息
                    3. 找不到任何实体时, 输出"没有找到任何实体".
                """
INSTRUCTION = """请提取如下句子的命名实体:"""


def get_tag_entities(tokens, tokens_tag) -> str:
    entities: list[str] = []

    token_text_stack = []  # 记录文字
    token_tags_stack = []  # 记录类别

    for token, token_tag in zip(tokens, tokens_tag):
        # token: 每一个token; token_tag: 每一个token的tag
        if token_tag.startswith('B-') or token_tag.startswith('I-'):
            token_text_stack.append(token)
            token_tags_stack.append(token_tag)
            continue

        if token_text_stack and token_tags_stack:
            entity_text = f'{''.join(token_text_stack)}'
            entity_label = f'{token_tags_stack[-1][2:]}'
            entities.append(f'{entity_label}:{entity_text}')
            token_text_stack = []
            token_tags_stack = []

    if token_text_stack and token_tags_stack:
        entity_text = f'{''.join(token_text_stack)}'
        entity_label = f'{token_tags_stack[-1][2:]}'
        entities.append(f'{entity_label}:{entity_text}')

    if not entities:
        return "未找到命名实体"
    return ';'.join(entities)


# 准备数据集
def prepare_dataset(texts, tokens_tags) -> Dataset:
    # 将文本拆分为字符列表
    tokens_lines = [list(text) for text in texts]
    tags = []
    for tokens, tokens_tag in zip(tokens_lines, tokens_tags):
        tags.append(get_tag_entities(tokens, tokens_tag))

    # 创建数据集
    dataset = Dataset.from_dict({
        "tokens": texts,
        "labels": tags
    })

    print(dataset.data)

    return dataset


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理数据"""
    # 加载训练数据
    train_lines = codecs.open('./msra/train/sentences.txt', encoding='utf-8').readlines()[:1000]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]

    train_tags = codecs.open('./msra/train/tags.txt', encoding='utf-8').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]

    train_dataset = prepare_dataset(train_lines, train_tags)

    # 加载验证数据
    val_lines = codecs.open('./msra/val/sentences.txt', encoding='utf-8').readlines()[:100]
    val_lines = [x.replace(' ', '').strip() for x in val_lines]

    val_tags = codecs.open('./msra/val/tags.txt', encoding='utf-8').readlines()[:100]
    val_tags = [x.strip().split(' ') for x in val_tags]

    val_dataset = prepare_dataset(val_lines, val_tags)

    return train_dataset, val_dataset


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
    instruction = tokenizer(
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{INSTRUCTION + example['tokens']}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )

    # 构建响应部分
    response = tokenizer(f"{example['labels']}", add_special_tokens=False)

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
        output_dir="./output_Qwen",
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

def parse_ner_response(response):
    """解析模型生成的NER响应"""
    entities = []
    response = response[:response.find("Human")]
    lines = response.split(';')

    for line in lines:
        line = line.strip()
        if ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                entity_type = parts[0].strip()
                entity_text = parts[1].strip()
                if entity_type in ['ORG', 'PER', 'LOC']:
                    entities.append((entity_text, entity_type))

    return entities

# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """命名实体识别"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION + text}
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
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # 解析响应
    entities = parse_ner_response(response)
    return entities, response


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """命名实体识别"""
    pred_labels = []

    for text in tqdm(test_texts, desc="命名实体识别"):
        try:
            pred_label, response = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 检查设备
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 1. 加载数据
    print("加载数据...")
    train_dataset, val_dataset = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "./models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    # tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(train_dataset.to_pandas())
    eval_ds = Dataset.from_pandas(val_dataset.to_pandas())

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
            pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # 评估模型
    print("评估模型...")
    eval_results = trainer.evaluate()
    print(f"评估结果: {eval_results}")

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_Qwen")


# 单独测试函数
def test_example():
    print("进行测试")
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model_path = "./models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_Qwen/")
    model = model.to(device)

    # 测试预测
    test_sentences = [
        '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',  # 人、位置
        '人工智能是未来的希望，也是中国和美国的冲突点。',
        '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
        '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
        '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
        '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
    ]

    # 预测
    preds = batch_predict(model, tokenizer, test_sentences)
    for pred in preds:
        print(pred)


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_example()
