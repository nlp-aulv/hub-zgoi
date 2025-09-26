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
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理数据"""
    train = json.load(open('E:/AI学习/Week07/cmrc2018_public/train.json', encoding='utf-8'))
    dev = json.load(open('E:/AI学习/Week07/cmrc2018_public/dev.json', encoding='utf-8'))

    def prepare_dataset(data, max_examples):
        examples = []
        for paragraph in data['data'][:max_examples]:
            context = paragraph['paragraphs'][0]['context']
            for qa in paragraph['paragraphs'][0]['qas']:
                examples.append({
                    "instruction": f"根据以下文本回答问题：{context}",  # 系统指令：提供上下文
                    "input": f"问题：{qa['question']}",  # 用户输入：具体问题
                    "output": f"答案：{qa['answers'][0]['text']}"  # 期望输出：正确答案
                })
        return examples

    train_examples = prepare_dataset(train,200)
    val_examples = prepare_dataset(dev,50)

    train_df = pd.DataFrame(train_examples)
    val_df = pd.DataFrame(val_examples)

    # 转换为Hugging Face Dataset格式
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    return train_ds, val_ds


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
def process_func(example, tokenizer, max_length=512):
    """
    处理单个样本的函数
    将指令和输出转换为模型训练格式
    """
    # 构建指令部分
    # ChatML 标准
    instruction_text = f"<|im_start|>system\n你是一个专业的知识问答助手，请根据提供的文本内容准确回答问题。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False)

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
        output_dir="E:/AI学习/Week07_xx/zy2_output_Qwen1.5",
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
def predict_intent(model, tokenizer, context, question, device='cpu'):
    """单个样本"""
    messages = [
        {"role": "system", "content": "你是一个专业的知识问答助手，请根据提供的文本内容准确回答问题。"},
        {"role": "user", "content": f"根据以下文本回答问题：{context}\n问题：{question}"}
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

    return response.strip()


# 批量预测
def batch_predict(model, tokenizer, context, question, device='cuda'):
    """批量预测测试集的答案"""
    pred_labels = []

    for context, question in tqdm(context, question, desc="知识问答"):
        try:
            pred_label = predict_intent(model, tokenizer, context, question, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"回答问题文'{question}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels


# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_ds,eval_ds = load_and_preprocess_data()

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "E:/AI学习/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    # tokenized_ds = ds.map(process_func_with_tokenizer, remove_columns=ds.column_names)

    # 4. 划分训练集和验证集
    train_ds = Dataset.from_pandas(train_ds.to_pandas())
    eval_ds = Dataset.from_pandas(eval_ds.to_pandas())

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

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model("E:/AI学习/Week07_xx/zy2_output_Qwen1.5/zy2_output_Qwen")
    tokenizer.save_pretrained("E:/AI学习/Week07_xx/zy2_output_Qwen1.5/zy2_output_Qwen")

# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "E:/AI学习/models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("E:/AI学习/Week07_xx/zy2_output_Qwen1.5/zy2_output_Qwen")
    model.cpu()

    # 测试预测
    context = "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品"
    question = "《战国无双3》是由哪两个公司合作开发的？"
    result = predict_intent(model, tokenizer, context, question)
    print(f"上下文: {context}")
    print(f"问题: {question}")
    print(f"答案: {result}")


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_single_example()