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
def load_and_preprocess_data(train_data_path,test_data_path,is_test=False):
    """加载和预处理数据"""
    with open(train_data_path, 'r', encoding='utf-8') as file:
        train_data = json.load(file)

    with open(test_data_path, 'r', encoding='utf-8') as file:
        test_data = json.load(file)

    # 格式化文本
    format_data = []

    # 划分训练集和验证集
    if is_test:
        # 测试集
        for data in test_data['data'][-100:]:
            context = data['paragraphs'][0]['context']
            for qas in data['paragraphs'][0]['qas']:
                question = qas['question']
                answers = qas['answers'][0]['text']

                insturction = f"请根据以下资料回答问题：{context}"

                format_data.append({
                    "instruction":insturction,
                    "input":question,
                    "output":answers
                })
    else:
        # 训练集
        for data in train_data['data'][:500]:
            context = data['paragraphs'][0]['context']
            for qas in data['paragraphs'][0]['qas']:
                question = qas['question']
                answers = qas['answers'][0]['text']

                insturction = f"请根据以下资料回答问题：{context}"

                format_data.append({
                    "instruction":insturction,
                    "input":question,
                    "output":answers
                })


    return format_data


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,  # 使用慢速但更稳定的tokenizer
        trust_remote_code=True  # 信任远程代码（自定义模型需要）
    )

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",          # 自动分配设备（GPU/CPU）
        torch_dtype=torch.float16  # 使用半精度减少内存占用
    )

    return tokenizer, model


# 数据处理函数
def process_func(example, tokenizer, max_length=384):

    instruction_text = f"<|im_start|>system\n你是一个阅读理解专家，需要从文本中找出问题对应的答案。<|im_end|>\n<|im_start|>user\n{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False) # 把上面的 instruction_text 转成 数字序列。

    # 构建响应部分
    response = tokenizer(f"{example['output']}", add_special_tokens=False) # 把训练数据里的 答案（output） 转成数字

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id] # 输入文本转成数字后的结果
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1] # 哪些位置是有效的文字，哪些是填充

    # 构建标签（指令部分用-100忽略，只计算响应部分的损失）
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id] # 告诉模型“你只需要学会生成答案部分（response），用户输入部分不用学”

    # 截断超过最大长度的序列
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]

    return {
        "input_ids": input_ids, # 文本向量化 模型的实际输入序列（prompt + 答案）
        "attention_mask": attention_mask, # 掩码，告诉模型哪些 token 有效，哪些是 padding
        "labels": labels # 训练目标，指明哪些位置需要计算 loss（通常只在答案部分）
    }


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], # 指定哪些模块的权重会被 LoRA 微调
        inference_mode=False, # 表示这是训练模式，LoRA 权重需要更新
        r=8, # r 越小，参数越少，越节省显存，但表示能力有限
        lora_alpha=32, # LoRA 的缩放系数，用于平衡 A*B 的幅度
        lora_dropout=0.1 # 在 LoRA 层输出前加 dropout，增加训练稳定性和正则化
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    return model


# 训练配置
def setup_training_args():
    """设置训练参数"""
    return TrainingArguments(
        output_dir="./output_qa_Qwen0.6",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        logging_steps=200,
        do_eval=True,
        eval_steps=500,
        num_train_epochs=1,
        save_steps=1000,
        learning_rate=5e-5,
        save_on_each_node=True,
        gradient_checkpointing=True,
        report_to="none"  # 禁用wandb等报告工具
    )


    # return TrainingArguments(
    #     output_dir="./output_qa_Qwen0.6",
    #     per_device_train_batch_size=2,
    #     gradient_accumulation_steps=8,
    #     logging_steps=100,
    #     do_eval=True,
    #     eval_steps=50,
    #     num_train_epochs=2,
    #     save_steps=50,
    #     learning_rate=5e-5,
    #     save_on_each_node=True,
    #     gradient_checkpointing=True,
    #     report_to="none"  # 禁用wandb等报告工具
    # )


# 预测函数
def predict_intent(model, tokenizer, text, question, device='cpu'):
    """预测单个文本的意图"""
    messages = [
        {"role": "system", "content": "你是一个阅读理解专家，需要从文本中找出问题对应的答案。"},
        {"role": "user", "content": f"请根据以下文本回答问题:{text}\n 问题:{question}"}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,  # 只格式化，不tokenize
        add_generation_prompt=True  # 添加助手提示符，让模型生成回复
    )

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,  # 最大生成token数
            do_sample=True,  # 使用采样而非贪婪搜索
            temperature=0.1,  # 低温度=更确定性输出
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

    for text in tqdm(test_texts, desc="预测意图"):
        try:
            pred_label = predict_intent(model, tokenizer, text, device)
            pred_labels.append(pred_label)
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            pred_labels.append("")  # 出错时添加空字符串

    return pred_labels



def convert_data(sentence, tag):
    data_list = []
    entities = []
    entity = ""
    label_BIO = ""

    for word,label in zip(sentence,tag):
        if label.startswith('B-'):
            if entity:
                entities.append((entity,label_BIO))
            entity = word
            label_BIO = label[2:]
        elif label.startswith('I-'):
            entity += word
        else:
            if entity:
                entities.append((entity, label_BIO))
            entity = ""
            label_BIO = ""

    if entities:
        entities_str = ",".join([f"'{entity}'({etype})" for entity,etype in entities])
        output = f"实体：{entities_str}"
    else:
        output = "未识别到实体"

    return {
        # "instruction": f"请从以下文本中识别出人名(PER)、地名(LOC)和组织名(ORG)实体：{sentence}",
        "instruction": sentence,
        "input": "",
        "output": output
    }


def format_data(sentences, tags):
    format_data_list=[]
    for sentence,tag in zip(sentences,tags):
        format_data_list.append(convert_data(sentence,tag))

    return format_data_list

# 主函数
def main():
    """主执行函数"""
    # 1. 加载数据
    print("加载数据...")
    train_data = load_and_preprocess_data('F:/developer_tools/github_desktop/Lfisfkmv_repo/hub-zgoi/杨仕雄/Week07/cmrc2018_public/train.json','F:/developer_tools/github_desktop/Lfisfkmv_repo/hub-zgoi/杨仕雄/Week07/cmrc2018_public/dev.json')
    test_data = load_and_preprocess_data('F:/developer_tools/github_desktop/Lfisfkmv_repo/hub-zgoi/杨仕雄/Week07/cmrc2018_public/train.json','F:/developer_tools/github_desktop/Lfisfkmv_repo/hub-zgoi/杨仕雄/Week07/cmrc2018_public/dev.json',is_test=True)

    # 普通数据 → DataFrame → Dataset
    train_dataset = Dataset.from_pandas(pd.DataFrame(train_data))
    val_dataset = Dataset.from_pandas(pd.DataFrame(test_data))

    # 2. 初始化模型和tokenizer
    print("初始化模型和tokenizer...")
    model_path = "F:/developer_tools/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 3. 处理数据
    print("处理训练数据...")
    process_func_with_tokenizer = lambda example: process_func(example, tokenizer)
    train_tokenized = train_dataset.map(process_func_with_tokenizer, remove_columns=train_dataset.column_names)
    eval_tokenized = val_dataset.map(process_func_with_tokenizer, remove_columns=val_dataset.column_names)

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
        model=model, # 训练模型
        args=training_args, # 训练参数
        train_dataset=train_tokenized, # 训练数据集
        eval_dataset=eval_tokenized,   # 验证数据集
        data_collator=DataCollatorForSeq2Seq( # 将不同长度的序列填充到相同长度
            tokenizer=tokenizer,
            padding=True, # 启用填充功能
            pad_to_multiple_of=8  # 优化GPU内存使用 填充长度是8的倍数
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./output_qa_Qwen0.6/")

# 单独测试函数
def test_single_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "F:/developer_tools/models/Qwen/Qwen3-0.6B"
    tokenizer, model = initialize_model_and_tokenizer(model_path)
    print("加载模型完毕")

    # 加载训练好的LoRA权重
    # model.load_adapter("F:/developer_tools/github_desktop/Lfisfkmv_repo/hub-zgoi/杨仕雄/Week07/model/output_Qwen1.5/")
    model.load_adapter("./output_qa_Qwen0.6/")
    model.cpu()
    print("加载权重完毕")

    test_text =  "冈察洛夫（俄语：，儒略历1812年6月6日－1891年9月15日，合格里历），俄罗斯小说家，代表作有《奥勃洛莫夫》（1859年）等。冈察洛夫在辛比尔斯克（今乌里扬诺夫斯克）出生，父亲是富裕的商人。1831年他进入莫斯科大学语文系，1834年大学毕业后到政府工作达30年。1847年，冈察洛夫创作的第一部小说《平凡的故事》出版,描写一个地主少爷顺应资本主义兴起的局势，成为一个实业家的故事，赢得了评论家别林斯基的好评。1849年他发表了中篇《奥勃洛莫夫的梦》。1852年至1855年间作为海军中将叶夫菲米·普佳京的秘书随他航行到英格兰、非洲和日本，后经西伯利亚返回俄罗斯。冈察洛夫据此旅途写作的游记《战舰“巴拉达”号》在1858年出版，翌年又发表长篇小说《奥勃洛莫夫》，大获好评。1867年，他辞去政府职务，并发表他最后一部长篇小说《悬崖》（1869年）。冈察洛夫终身未婚，1891年在圣彼得堡逝世。"
    question = "冈察洛夫是哪一国的小说家？"
    result = predict_intent(model, tokenizer, test_text,question)
    print(f"输入: {test_text}")
    print(f"问题: {question}")
    print(f"期望答案: 俄罗斯")
    print(f"实际预测答案: {result}")


if __name__ == "__main__":
    # 执行主函数
    # result_df = main()

    # 单独测试
    test_single_example()