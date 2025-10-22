import codecs
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from tqdm import tqdm
import torch

import warnings
warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}

# 加载训练数据
train_lines = codecs.open('E:/AI学习/Week07/msra/train/sentences.txt', encoding="utf-8").readlines()[:1000]
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags = codecs.open('E:/AI学习/Week07/msra/train/tags.txt', encoding="utf-8").readlines()[:1000]
train_tags = [x.strip().split(' ') for x in train_tags]
train_tags_id = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = codecs.open('E:/AI学习/Week07/msra/val/sentences.txt', encoding="utf-8").readlines()[:100]
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags = codecs.open('E:/AI学习/Week07/msra/val/tags.txt', encoding="utf-8").readlines()[:100]
val_tags = [x.strip().split(' ') for x in val_tags]
val_tags_id = [[label2id[x] for x in tag] for tag in val_tags]

def load_and_preprocess_data(texts, tags):
    """加载和预处理数据"""
    tags = [' '.join(tag) for tag in tags]
    ds = Dataset.from_dict({"instruction": texts, "output": tags})

    # 添加输入列
    ds = ds.add_column(name='input', column=[""] * len(ds))
    return ds

# 初始化Qwen模型和对应的分词器
def initialize_model_and_tokenizer(model_path):
    """初始化Qwen模型和对应的分词器"""
    # 加载Qwen的分词器
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
def process_func(example, tokenizer, max_length=256):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n你是一个中文命名实体识别专家。给定一个中文句子，为每个字输出对应的BIO标签。标签说明：B-PER(人名开始), I-PER(人名中间), B-ORG(组织开始), I-ORG(组织中间), B-LOC(地点开始), I-LOC(地点中间), O(非实体)。输出格式：标签1 标签2 标签3...<|im_end|>\n"
    instruction_text += f"<|im_start|>user\n文本：{example['instruction']}<|im_end|>\n"
    instruction_text += f"<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

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
        lora_alpha=16,
        lora_dropout=0.1
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="E:/AI学习/Week07_xx/zy1_output_Qwen1.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        do_eval=True,
        eval_steps=20,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


# 预测函数
def predict_intent(model, tokenizer, text, device='cpu'):
    """识别单个文本"""
    model.eval()
    messages = [
        {"role": "system", "content": "你是一个中文命名实体识别专家。输出BIO标签序列，每个标签必须是：O, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC 中的一个。"},
        {"role": "user", "content": f"文本：{text}"}
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
            max_new_tokens=150,
            do_sample=True,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 批量预测
def batch_predict(model, tokenizer, test_texts, device='cuda'):
    """批量预测测试集的意图"""
    pred_labels = []

    for text in tqdm(test_texts, desc="实体识别"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"识别文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds = load_and_preprocess_data(train_lines, train_tags)
    val_ds = load_and_preprocess_data(val_lines, val_tags)

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "E:/AI学习/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    # tokenized_ds = ds.map(process_func_with_tokenizer,  remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(train_ds.to_pandas())
    eval_ds = Dataset.from_pandas(val_ds.to_pandas())

    train_tokenized = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    eval_tokenized = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)
    for i in range(3):
        print(f"\n样本 {i}:")
        print(f"句子: '{train_lines[i]}'")
        print(f"标签: {train_tags[i]}")
        print(f"句子字符数: {len(train_lines[i])}")
        print(f"标签数量: {len(train_tags[i])}")
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
    print("训练完成")
    # 8. 保存模型
    print("保存模型...")
    output_model_path = "E:/AI学习/Week07_xx/zy1_output_Qwen1.5/zy1_output_Qwen"
    trainer.save_model(output_model_path)
    tokenizer.save_pretrained(output_model_path)

# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "E:/AI学习/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model = PeftModel.from_pretrained(model, "E:/AI学习/Week07_xx/zy1_output_Qwen1.5/zy1_output_Qwen")
    model.cpu()

    # 测试预测
    test_text = '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。'
    result = predict_intent(model, tokenizer, test_text)
    print(f"输入: {test_text}")
    print(f"实体: {result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_single_example()