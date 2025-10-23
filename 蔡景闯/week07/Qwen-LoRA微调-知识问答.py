import codecs
import json
import os
import shutil

import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)
# pip install peft
from peft import LoraConfig, TaskType, get_peft_model
from tqdm import tqdm
import torch


# 数据加载和预处理
def load_and_preprocess_data(tokenizer):
    # 加载数据
    train = json.load(open('./cmrc2018_public/train.json', encoding="utf-8"))
    dev = json.load(open('./cmrc2018_public/dev.json', encoding="utf-8"))
    # 准备训练和验证数据
    train_paragraphs, train_questions, train_answers = prepare_dataset(train)
    val_paragraphs, val_questions, val_answers = prepare_dataset(dev)
    # 创建数据集字典
    train_dataset_dict = {
        'context': train_paragraphs[:1000],
        'question': train_questions[:1000],
        'answers': train_answers[:1000]
    }

    val_dataset_dict = {
        'context': val_paragraphs[:100],
        'question': val_questions[:100],
        'answers': val_answers[:100]
    }
    # 转换为Hugging Face Dataset
    train_dataset = Dataset.from_dict(train_dataset_dict)
    val_dataset = Dataset.from_dict(val_dataset_dict)

    # 应用预处理
    tokenized_train_dataset = train_dataset.map(
        lambda exp: preprocess_function(exp, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
    )

    tokenized_val_dataset = val_dataset.map(
        lambda exp: preprocess_function(exp, tokenizer),
        batched=True,
        remove_columns=val_dataset.column_names,
    )
    return tokenized_train_dataset, tokenized_val_dataset


def preprocess_function(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 由于可能有溢出，需要重新映射样本
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    tokenized_examples["start_positions"] = []
    tokenized_examples["end_positions"] = []

    for i, offsets in enumerate(offset_mapping):
        # 获取对应的原始样本
        sample_index = sample_mapping[i]
        answer = examples["answers"][sample_index]

        # 如果没有答案，设置默认值
        if len(answer["answer_start"]) == 0:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 找到token的起始和结束位置
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 找到context的开始和结束
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # 如果答案完全在context之外，标记为不可回答
        if offset_mapping[i][context_start][0] > end_char or offset_mapping[i][context_end][1] < start_char:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # 否则找到答案的token位置
            idx = context_start
            while idx <= context_end and offset_mapping[i][idx][0] <= start_char:
                idx += 1
            start_position = idx - 1

            idx = context_end
            while idx >= context_start and offset_mapping[i][idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


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
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16
    )

    return tokenizer, model


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置并应用到模型"""
    config = LoraConfig(
        task_type=TaskType.QUESTION_ANS,
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
def setup_training_args(dir):
    """设置训练参数"""
    return TrainingArguments(
        output_dir=dir,
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
def predict_answer(model, tokenizer, question, context, device='cuda'):
    """预测答案"""
    # Tokenize 输入
    inputs = tokenizer(
        question,
        context,
        truncation=True,
        max_length=512,
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs.to(device))
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    # 提取答案
    start_index = start_logits.argmax().item()
    end_index = end_logits.argmax().item()

    # 解码答案
    answer_tokens = inputs.input_ids[0][start_index:end_index + 1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

    return answer


# 主函数
def main():
    """主执行函数"""
    dir = "./output_Qwen1.5_QA"
    # 清空输出目录
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir, exist_ok=True)

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
    training_args = setup_training_args(dir)

    # 创建Trainer并开始训练
    print("开始训练...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=DefaultDataCollator(),
    )

    trainer.train()

    # 保存模型
    print("保存模型和 LoRA 权重...")
    model.save_pretrained(dir)  # 保存 LoRA 权重
    tokenizer.save_pretrained(dir)  # 保存分词器


# 单独测试函数
def test_single_example(dir):
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "../models/Qwen/Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter(dir)
    model.to('cuda')

    # 测试预测
    test_question = "罗亚尔港号是什么级别的导弹巡洋舰？"
    test_context = "罗亚尔港号（USS Port Royal CG-73）是美国海军提康德罗加级导弹巡洋舰，是该级巡洋舰的第27艘也是最后一艘。它也是美国海军第二艘以皇家港为名字命名的军舰。第一艘是1862年下水、曾参与南北战争的。船名来自曾在美国独立战争和南北战争中均发生过海战的南卡罗来纳州（Port Royal Sound）。美国海军在1988年2月25日订购该船，1991年10月18日在密西西比州帕斯卡古拉河畔的英戈尔斯造船厂放置龙骨。1992年11月20日下水，1992年12月5日由苏珊·贝克（Susan G. Baker，老布什政府时期的白宫办公厅主任，也是前国务卿詹姆斯·贝克的夫人）为其命名，1994年7月9日正式服役。2009年2月5日，罗亚尔港号巡洋舰在位于檀香山国际机场以南0.5英里的一处珊瑚礁上发生搁浅，之前该舰刚完成在旱坞内的维护，正在进行维护后的第一次海试。2009年2月9日凌晨2点，罗亚尔港号被脱离珊瑚礁。无人在这次事故中受伤，也未发生船上燃料的泄漏。但由于这次搁浅，罗亚尔港号巡洋舰不得不回到旱坞重新进行维修。1995年12月加入尼米兹号为核心的航空母舰战斗群，参与了南方守望行动，这是罗亚尔港号巡洋舰首次参与的军事部署行动。1996年3月由于台湾海峡导弹危机的发生被部署到了南中国海，随着危机的结束，1997年9月至1998年3月回到尼米兹号航空母舰战斗群参与南方守望行动。后随约翰·C·斯坦尼斯号航空母舰战斗群继续参加南方守望行动。2000年1月由于多次追击涉嫌违反联合国禁运制裁走私偷运伊拉克原油的船只因而造成对船上动力设备的持续性机械磨损而撤离，回到夏威夷进行整修和升级。2001年11月7日加入约翰·C·斯坦尼斯号航空母舰战斗群参与旨在对基地组织和对它进行庇护的阿富汗塔利班政权进行打击的持久自由军事行动。"
    result = predict_answer(model, tokenizer, test_question, test_context)
    print(f"问题: {test_question}")
    print(f"上下文: {test_context}")
    print(f"答案: {result}")


if __name__ == "__main__":
    # 执行主函数
    main()

    # 单独测试
    test_single_example("./output_Qwen1.5_QA")
