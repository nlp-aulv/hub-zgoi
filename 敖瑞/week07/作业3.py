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


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理JSON格式的问答数据"""
    with open('trial.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    instructions = []
    outputs = []
    inputs = []

    # 解析SQuAD格式的JSON数据
    for article in data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                question = qa['question']
                # 取第一个答案作为训练目标
                if qa['answers']:
                    answer = qa['answers'][0]['text']
                else:
                    answer = ""

                # 构建输入格式：上下文 + 问题
                instructions.append(f"Context: {context} Question: {question}")
                outputs.append(answer)
                inputs.append("")  # 保持原始结构，但input字段为空

    train_data = pd.DataFrame({
        "instruction": instructions,
        "output": outputs,
        "input": inputs
    })

    train_data = Dataset.from_pandas(train_data)

    print(f"数据数量: {len(train_data)}")
    print("示例数据:")
    for i in range(min(2, len(train_data))):
        print(f"指令: {train_data[i]['instruction']}")
        print(f"输入: {train_data[i]['input']}")
        print(f"输出: {train_data[i]['output']}")
        print("---")

    return train_data


# 初始化tokenizer和模型
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
        torch_dtype=torch.float16
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=512):
    """知识问答格式"""
    instruction_text = f"<|im_start|>system\n现在进行知识问答任务<|im_end|>\n<|im_start|>user\n{example['instruction'] + example[
        'input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    response = tokenizer(f"{example['output']}<|im_end|>", add_special_tokens=False)

    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"]

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


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./qa_output_Qwen1.5",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=50,
        do_eval=True,
        eval_steps=50,
        num_train_epochs=5,
        save_steps=200,
        learning_rate=1e-4,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none",
        prediction_loss_only=True,
        remove_unused_columns=False
    )


# 知识问答预测函数
def predict_qa(model, tokenizer, context, question, device='cpu'):
    """预测单个问答对"""
    input_text = f"Context: {context} Question: {question}"

    messages = [
        {"role": "system", "content": "现在进行知识问答任务，请根据提供的文本回答问题"},
        {"role": "user", "content": input_text}
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
            max_new_tokens=128,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )

    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip()


# 批量预测函数
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
    print("加载数据...")
    ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "./Qwen3-0.6B"  # 根据实际路径调整
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    num = int(0.8 * len(ds))
    train_ds = tokenized_ds.select(range(num))
    eval_ds = tokenized_ds.select(range(num, len(ds)))

    print(f"训练集大小: {len(train_ds)}")
    print(f"验证集大小: {len(eval_ds)}")

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
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            pad_to_multiple_of=8
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./qa_output_Qwen")


# 测试函数
def test_qa_example():
    """测试单个问答示例"""
    model_path = "./Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./qa_output_Qwen1.5/")
    model.cpu()

    # 从trial.json中加载测试数据
    with open("trial.json", 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 示例
    context = '《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品'
    question = '《战国无双3》是由哪两个公司合作开发的'
    true_answer = '光荣和ω-force'

    print(f"上下文: {context}")
    print(f"问题: {question}")
    print(f"真实答案: {true_answer}")

    # 预测
    print('预测中...')
    pred_answer = predict_qa(model, tokenizer, context, question, device='cpu')
    print(f"预测答案: {pred_answer}")


if __name__ == "__main__":
    # 执行训练
    main()

    # 测试单个示例
    test_qa_example()
