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
from tqdm import tqdm
import torch
import json


# 数据加载和预处理 - 修改为处理分开的问题和答案
def load_and_preprocess_data():
    """加载和预处理QA数据"""

    # 加载数据文件
    with open('./cmrc2018_public/train.json', 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    with open('./cmrc2018_public/dev.json', 'r', encoding='utf-8') as f:
        val_data = json.load(f)

    # 提取训练数据
    train_qa_data = []

    for item in train_data['data'][:1000]:  # 限制数据量
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # 每个问题只有一个答案
                answer = qa['answers'][0]['text']  # 直接取答案文本

                instruction = f"请根据以下文本回答问题：{context}"
                input_text = f"问题：{question}"

                train_qa_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": answer
                })

    # 提取验证数据
    val_qa_data = []

    for item in val_data['data'][:100]:  # 限制数据量
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                answer = qa['answers'][0]['text']  # 直接取答案文本

                instruction = f"请根据以下文本回答问题：{context}"
                input_text = f"问题：{question}"

                val_qa_data.append({
                    "instruction": instruction,
                    "input": input_text,
                    "output": answer
                })

    # 创建训练和验证Dataset
    train_df = pd.DataFrame(train_qa_data)
    val_df = pd.DataFrame(val_qa_data)

    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    print(f"训练数据数量: {len(train_ds)}")
    print(f"验证数据数量: {len(val_ds)}")
    print("示例数据:")
    for i in range(min(2, len(train_ds))):
        print(f"指令: {train_ds[i]['instruction']}")
        print(f"输入: {train_ds[i]['input']}")
        print(f"输出: {train_ds[i]['output']}")
        print("---")

    return train_ds, val_ds


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    """
    # 构建指令部分
    instruction_text = f"<|im_start|>system\n现在进行阅读理解问答任务，请根据提供的文本回答问题<|im_end|>\n<|im_start|>user\n{example['instruction']}\n{example['input']}<|im_end|>\n<|im_start|>assistant\n"
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


# 初始化tokenizer和模型（
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )

    return tokenizer, model


# 配置LoRA（保持不变）
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
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


# 训练配置（保持不变）
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_qa_qwen",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=100,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        prediction_loss_only=True,
        remove_unused_columns=False
    )


# 预测函数保持不变
def predict_qa(model, tokenizer, context, question, device='cpu'):
    """预测问答结果"""
    messages = [
        {"role": "system", "content": "现在进行阅读理解问答任务，请根据提供的文本回答问题"},
        {"role": "user", "content": f"请根据以下文本回答问题：{context}\n问题：{question}"}
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
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response


# 批量预测QA
def batch_predict_qa(model, tokenizer, test_contexts, test_questions, test_answers=None, device='cuda'):
    """批量预测QA"""
    results = []

    for i in tqdm(range(len(test_contexts)), desc="预测问答"):
        try:
            context = test_contexts[i]
            question = test_questions[i]
            true_answer = test_answers[i] if test_answers else None

            predicted_answer = predict_qa(model, tokenizer, context, question, device)

            results.append({
                "context": context,
                "question": question,
                "true_answer": true_answer,
                "predicted_answer": predicted_answer
            })
        except Exception as e:
            print(f"预测时出错: {e}")
            results.append({
                "context": test_contexts[i] if i < len(test_contexts) else '',
                "question": test_questions[i] if i < len(test_questions) else '',
                "true_answer": test_answers[i] if test_answers and i < len(test_answers) else '',
                "predicted_answer": ''
            })

    return results


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载QA数据...")

    # 加载果JSON文件
    train_ds, eval_ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "./models/Qwen/Qwen3-0___6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_train_ds = train_ds.map(process_func_with_tokenizer, remove_columns=train_ds.column_names)
    tokenized_eval_ds = eval_ds.map(process_func_with_tokenizer, remove_columns=eval_ds.column_names)

    print(f"训练集大小: {len(tokenized_train_ds)}")
    print(f"验证集大小: {len(tokenized_eval_ds)}")

    # 4. 设置LoRA
    print("设置LoRA...")
    model.enable_input_require_grads()
    model = setup_lora(model)

    # 5. 配置训练参数
    print("配置训练参数...")
    training_args = setup_training_args()

    # 6. 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # 7. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_qa_qwen")

    return trainer


# 测试函数
def test_qa():
    """测试QA功能"""
    model_path = "./models/Qwen/Qwen3-0___6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./output_qa_qwen/")
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试数据 - 使用分开的格式
    test_contexts = [
        "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。",
        "范廷颂枢机（，），圣名保禄·若瑟（），是越南罗马天主教枢机。1963年被任为主教；1990年被擢升为天主教河内总教区宗座署理；1994年被擢升为总主教，同年年底被擢升为枢机；2009年2月离世。"
    ]

    test_questions = [
        "范廷颂是什么时候被任为主教的？",
        "1990年，范廷颂担任什么职务？"
    ]

    test_answers = [
        "1963年",
        "1990年被擢升为天主教河内总教区宗座署理"
    ]

    print("开始QA测试...")
    results = batch_predict_qa(model, tokenizer, test_contexts, test_questions, test_answers)

    for result in results:
        print(f"\n上下文: {result['context'][:100]}...")
        print(f"问题: {result['question']}")
        print(f"真实答案: {result['true_answer']}")
        print(f"预测答案: {result['predicted_answer']}")
        print("-" * 80)


if __name__ == "__main__":
    # 执行主函数
    trainer = main()

    # 测试QA功能
    test_qa()