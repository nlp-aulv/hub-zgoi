import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    DefaultDataCollator
)
from datasets import Dataset
import numpy as np
from peft import LoraConfig, TaskType, get_peft_model

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 加载数据
train = json.load(open('./cmrc2018_public/train.json', 'r', encoding='utf-8'))
dev = json.load(open('./cmrc2018_public/dev.json', 'r', encoding='utf-8'))

# 初始化Qwen tokenizer和模型
model_path = "../models/Qwen/Qwen3-0.6B"  # 可以根据需要调整模型大小
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForQuestionAnswering.from_pretrained(
    model_path,
    trust_remote_code=True,
    torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32
)

# 如果tokenizer没有pad_token，设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置"""
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # 对于问答任务使用序列分类任务类型
        inference_mode=False,
        r=16,  # LoRA秩
        lora_alpha=32,  # LoRA缩放因子
        lora_dropout=0.1,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],  # Qwen模型的常见目标模块
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


# 应用LoRA到模型
print("应用LoRA配置...")
model = setup_lora(model)
model.to(device)


# 准备训练数据
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


# 预处理函数 - 适配Qwen模型
def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    contexts = [c.strip() for c in examples["context"]]

    # Tokenize - 适配Qwen的格式
    tokenized_examples = tokenizer(
        questions,
        contexts,
        truncation="only_second",  # 只截断context部分
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
        if len(answer["answer_start"]) == 0 or not answer["text"]:
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
            continue

        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        # 找到token的起始和结束位置
        sequence_ids = tokenized_examples.sequence_ids(i)

        # 找到context的开始和结束
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx

        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1 if idx > 0 else len(sequence_ids) - 1

        # 如果答案完全在context之外，标记为不可回答
        if (context_start >= len(offsets) or context_end >= len(offsets) or
                offsets[context_start][0] > end_char or offsets[context_end][1] < start_char):
            tokenized_examples["start_positions"].append(0)
            tokenized_examples["end_positions"].append(0)
        else:
            # 否则找到答案的token位置
            idx = context_start
            while idx <= context_end and idx < len(offsets) and offsets[idx][0] <= start_char:
                idx += 1
            start_position = idx - 1 if idx > context_start else context_start

            idx = context_end
            while idx >= context_start and idx < len(offsets) and offsets[idx][1] >= end_char:
                idx -= 1
            end_position = idx + 1 if idx < context_end else context_end

            # 确保位置有效
            start_position = max(context_start, min(start_position, context_end))
            end_position = max(context_start, min(end_position, context_end))

            tokenized_examples["start_positions"].append(start_position)
            tokenized_examples["end_positions"].append(end_position)

    return tokenized_examples


# 应用预处理
print("预处理训练数据...")
tokenized_train_dataset = train_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=train_dataset.column_names,
)

print("预处理验证数据...")
tokenized_val_dataset = val_dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=val_dataset.column_names,
)

print(f"训练集大小: {len(tokenized_train_dataset)}")
print(f"验证集大小: {len(tokenized_val_dataset)}")

# 设置训练参数 - 针对Qwen和LoRA优化
training_args = TrainingArguments(
    output_dir="./qwen-qa-lora-model",
    learning_rate=2e-4,  # LoRA通常使用更高的学习率
    per_device_train_batch_size=4,  # 根据GPU内存调整
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=200,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    report_to="none",
    gradient_accumulation_steps=2,
    warmup_steps=100,
    fp16=(device.type == 'cuda'),  # 在CUDA上使用混合精度训练
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# 数据收集器
data_collator = DefaultDataCollator()

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# 开始训练
print("开始Qwen-LoRA问答模型训练...")
trainer.train()

# 保存模型（包括LoRA适配器）
trainer.save_model()
tokenizer.save_pretrained('./qwen-qa-lora-model')
print("模型保存完成")

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")


# 预测函数 - 适配Qwen模型
def predict(context, question, model, tokenizer):
    """使用Qwen-LoRA模型进行问答预测"""
    model.eval()

    # Tokenize输入
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    ).to(model.device)

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    # 获取预测的起始和结束位置
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

    # 找到最可能的答案跨度
    start_idx = torch.argmax(start_logits, dim=1).item()
    end_idx = torch.argmax(end_logits, dim=1).item()

    # 将token位置转换为文本
    input_ids = inputs["input_ids"][0].cpu().numpy()

    # 提取答案的token IDs
    answer_token_ids = input_ids[start_idx:end_idx + 1]

    # 将token转换回文本
    answer = tokenizer.decode(answer_token_ids, skip_special_tokens=True)

    # 清理答案（针对中文）
    answer = answer.strip()

    return answer


# 在验证集上测试几个样本
print("\n在验证集上测试:")
for i in range(min(5, len(val_paragraphs))):
    context = val_paragraphs[i]
    question = val_questions[i]
    expected_answer = val_answers[i]['text'][0] if val_answers[i]['text'] else "无答案"

    predicted_answer = predict(context, question, model, tokenizer)

    print(f"问题 {i + 1}: {question}")
    print(f"上下文: {context[:100]}...")
    print(f"预期答案: {expected_answer}")
    print(f"预测答案: {predicted_answer}")
    print(f"匹配: {expected_answer == predicted_answer}")
    print("-" * 80)


# 加载和使用训练好的LoRA模型
def load_trained_qa_model(base_model_path, adapter_path):
    """加载训练好的Qwen-LoRA问答模型"""
    # 加载基础模型
    base_model = AutoModelForQuestionAnswering.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )

    # 加载LoRA配置和权重
    model = get_peft_model(base_model, LoraConfig.from_pretrained(adapter_path))
    model.load_adapter(adapter_path)

    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)

    return model, tokenizer


# 测试加载的模型
def test_loaded_model():
    """测试加载的训练好的模型"""
    try:
        print("测试加载的模型...")
        loaded_model, loaded_tokenizer = load_trained_qa_model(
            model_path,
            './qwen-qa-lora-model/'
        )
        loaded_model.to(device)

        # 测试样例
        test_context = "北京大学创建于1898年，是中国最早的现代国立大学之一。"
        test_question = "北京大学创建于哪一年？"

        answer = predict(test_context, test_question, loaded_model, loaded_tokenizer)
        print(f"测试问题: {test_question}")
        print(f"测试上下文: {test_context}")
        print(f"预测答案: {answer}")

    except Exception as e:
        print(f"加载模型时出错: {e}")


# 运行测试
test_loaded_model()


# 计算准确率的函数
def calculate_accuracy(model, tokenizer, contexts, questions, answers, num_samples=50):
    """计算模型在测试集上的准确率"""
    model.eval()
    correct = 0
    total = min(num_samples, len(contexts))

    for i in range(total):
        try:
            context = contexts[i]
            question = questions[i]
            expected_answer = answers[i]['text'][0] if answers[i]['text'] else ""

            predicted_answer = predict(context, question, model, tokenizer)

            # 简单的字符串匹配（可以根据需要改进）
            if expected_answer.strip() == predicted_answer.strip():
                correct += 1

        except Exception as e:
            print(f"处理第 {i} 个样本时出错: {e}")
            continue

    accuracy = correct / total if total > 0 else 0
    return accuracy


# 计算验证集准确率
print("\n计算验证集准确率...")
val_accuracy = calculate_accuracy(model, tokenizer, val_paragraphs, val_questions, val_answers, 50)
print(f"验证集准确率: {val_accuracy:.4f}")

print("训练和测试完成！")