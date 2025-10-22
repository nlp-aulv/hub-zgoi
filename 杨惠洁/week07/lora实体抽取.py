import codecs
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, classification_report
import warnings
from peft import LoraConfig, TaskType, get_peft_model

warnings.filterwarnings('ignore')

# 检查设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义标签类型
tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in enumerate(tag_type)}


# 加载数据函数
def load_data(file_path, max_lines=None):
    """加载数据文件，尝试多种编码"""
    encodings = ['utf-8', 'gbk', 'utf-8-sig', 'latin-1']
    for encoding in encodings:
        try:
            with codecs.open(file_path, encoding=encoding) as f:
                lines = f.readlines()
                if max_lines:
                    lines = lines[:max_lines]
                return [line.strip() for line in lines]
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法解码文件: {file_path}")


# 加载训练数据
train_lines = load_data('./msra/train/sentences.txt', 1000)
train_lines = [x.replace(' ', '').strip() for x in train_lines]

train_tags_lines = load_data('./msra/train/tags.txt', 1000)
train_tags = [x.strip().split(' ') for x in train_tags_lines]
train_tags = [[label2id[x] for x in tag] for tag in train_tags]

# 加载验证数据
val_lines = load_data('./msra/val/sentences.txt', 100)
val_lines = [x.replace(' ', '').strip() for x in val_lines]

val_tags_lines = load_data('./msra/val/tags.txt', 100)
val_tags = [x.strip().split(' ') for x in val_tags_lines]
val_tags = [[label2id[x] for x in tag] for tag in val_tags]

# 初始化Qwen tokenizer
model_path = "../models/Qwen/Qwen2.5-0.5B"  # 可以根据需要调整模型大小
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 如果tokenizer没有pad_token，设置为eos_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 对数据进行tokenize和标签对齐
def tokenize_and_align_labels(examples):
    # 将字符列表转换为字符串（Qwen tokenizer需要字符串输入）
    texts = [''.join(tokens) for tokens in examples["tokens"]]

    tokenized_inputs = tokenizer(
        texts,
        truncation=True,
        padding=False,
        max_length=256,
        return_offsets_mapping=True,  # 用于字符到token的映射
        is_split_into_words=False,  # 输入是完整字符串
    )

    labels = []
    for i, (text, label) in enumerate(zip(texts, examples["labels"])):
        # 获取字符级别的标签
        char_labels = label  # 已经是字符级别的标签列表

        # 获取offset mapping
        offset_mapping = tokenized_inputs["offset_mapping"][i]

        label_ids = []
        for offset_idx, (start, end) in enumerate(offset_mapping):
            # 特殊token设置为-100
            if start == end == 0:
                label_ids.append(-100)
                continue

            # 获取当前token对应的字符位置
            if start < len(char_labels):
                # 取第一个字符的标签（对于中文，通常一个token对应一个字符）
                label_ids.append(char_labels[start])
            else:
                # 超出文本范围，设置为-100
                label_ids.append(-100)

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# 准备数据集
def prepare_dataset(texts, tags):
    tokens = [list(text) for text in texts]

    dataset = Dataset.from_dict({
        "tokens": tokens,
        "labels": tags
    })

    tokenized_dataset = dataset.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=dataset.column_names
    )

    return tokenized_dataset


# 准备训练和验证数据集
train_dataset = prepare_dataset(train_lines, train_tags)
eval_dataset = prepare_dataset(val_lines, val_tags)

print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(eval_dataset)}")


# 加载Qwen模型并进行修改以支持序列标注
def load_model_for_token_classification(model_path, num_labels):
    """加载Qwen模型并修改为序列标注任务"""
    model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        trust_remote_code=True
    )

    # 修改模型配置以支持序列标注
    model.config.pad_token_id = tokenizer.pad_token_id

    return model


# 加载模型
print("加载Qwen模型...")
model = load_model_for_token_classification(model_path, len(tag_type))


# 配置LoRA
def setup_lora(model):
    """设置LoRA配置"""
    config = LoraConfig(
        task_type=TaskType.TOKEN_CLS,
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

# 设置训练参数
training_args = TrainingArguments(
    output_dir='./qwen-ner-lora-model',
    learning_rate=5e-4,  # Qwen模型适合的学习率
    per_device_train_batch_size=8,  # 根据GPU内存调整
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    logging_dir='./logs',
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
    fp16=(device.type == 'cuda'),
    dataloader_pin_memory=False,
    remove_unused_columns=False,
)

# 数据收集器
data_collator = DataCollatorForTokenClassification(
    tokenizer=tokenizer,
    padding=True
)


# 定义计算指标的函数
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        valid_preds = []
        valid_labels = []
        for p, l in zip(prediction, label):
            if l != -100:  # 忽略特殊token
                valid_preds.append(id2label[p])
                valid_labels.append(id2label[l])
        true_predictions.append(valid_preds)
        true_labels.append(valid_labels)

    flat_true_predictions = [item for sublist in true_predictions for item in sublist]
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    if len(flat_true_labels) == 0:
        return {"accuracy": 0, "macro_f1": 0}

    accuracy = accuracy_score(flat_true_labels, flat_true_predictions)

    report = classification_report(
        flat_true_labels,
        flat_true_predictions,
        output_dict=True,
        zero_division=0
    )

    f1_scores = {}
    for label in tag_type:
        if label in report:
            f1_scores[f"{label}_f1"] = report[label]["f1-score"]

    macro_f1 = report['macro avg']['f1-score'] if 'macro avg' in report else 0
    weighted_f1 = report['weighted avg']['f1-score'] if 'weighted avg' in report else 0

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        **f1_scores
    }


# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# 开始训练
print("开始Qwen-LoRA命名实体识别训练...")
trainer.train()

# 保存模型（包括LoRA适配器）
trainer.save_model()
tokenizer.save_pretrained('./qwen-ner-lora-model')
print("模型保存完成")

# 评估模型
print("评估模型...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")


# 预测函数
def predict_ner(sentence, model, tokenizer):
    """使用Qwen-LoRA模型进行命名实体识别"""
    # 将模型设置为评估模式
    model.eval()

    # Tokenize输入
    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=256,
        return_offsets_mapping=True
    ).to(model.device)

    # 预测
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)
    predicted_labels = [id2label[p.item()] for p in predictions[0]]

    # 获取原始文本的字符列表
    chars = list(sentence)

    # 使用offset mapping对齐标签
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    aligned_labels = []

    for i, (start, end) in enumerate(offset_mapping):
        if start == end == 0:  # 特殊token
            continue
        if start < len(chars):
            aligned_labels.append(predicted_labels[i])

    # 确保标签数量与字符数量一致
    if len(aligned_labels) > len(chars):
        aligned_labels = aligned_labels[:len(chars)]
    elif len(aligned_labels) < len(chars):
        aligned_labels.extend(['O'] * (len(chars) - len(aligned_labels)))

    # 提取实体
    entities = []
    current_entity = ""
    current_type = ""

    for char, label in zip(chars, aligned_labels):
        if label.startswith('B-'):
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = char
            current_type = label[2:]
        elif label.startswith('I-') and current_entity and current_type == label[2:]:
            current_entity += char
        else:
            if current_entity:
                entities.append((current_entity, current_type))
            current_entity = ""
            current_type = ""
            if label.startswith('B-'):
                current_entity = char
                current_type = label[2:]

    if current_entity:
        entities.append((current_entity, current_type))

    return entities, aligned_labels


# 测试预测
test_sentences = [
    '今天我约了王浩在恭王府吃饭，晚上在天安门逛逛。',
    '人工智能是未来的希望，也是中国和美国的冲突点。',
    '明天我们一起在海淀吃个饭吧，把叫刘涛和王华也叫上。',
    '同煤集团同生安平煤业公司发生井下安全事故 19名矿工遇难',
    '山东省政府办公厅就平邑县玉荣商贸有限公司石膏矿坍塌事故发出通报',
    '[新闻直播间]黑龙江:龙煤集团一煤矿发生火灾事故'
]

print("\n测试预测结果:")
for sentence in test_sentences:
    try:
        entities, labels = predict_ner(sentence, model, tokenizer)
        print(f"句子: {sentence}")
        print(f"字符标签: {''.join([f'{char}[{label}]' for char, label in zip(sentence, labels)])}")
        if entities:
            for entity, entity_type in entities:
                print(f"  {entity_type}: {entity}")
        else:
            print("  未识别到实体")
        print()
    except Exception as e:
        print(f"处理句子时出错: {sentence}")
        print(f"错误信息: {e}")
        print()


# 加载和使用训练好的LoRA模型
def load_trained_model(model_path, adapter_path):
    """加载训练好的Qwen-LoRA模型"""
    # 加载基础模型
    base_model = AutoModelForTokenClassification.from_pretrained(
        model_path,
        num_labels=len(tag_type),
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=True
    )

    # 加载LoRA配置和权重
    model = get_peft_model(base_model, LoraConfig.from_pretrained(adapter_path))
    model.load_adapter(adapter_path)

    return model


# 测试加载的模型
def test_loaded_model():
    """测试加载的训练好的模型"""
    try:
        print("测试加载的模型...")
        loaded_model = load_trained_model(model_path, './qwen-ner-lora-model/')
        loaded_model.to(device)

        test_text = "北京大学位于北京市海淀区，张三教授在那里工作。"
        entities, labels = predict_ner(test_text, loaded_model, tokenizer)
        print(f"测试句子: {test_text}")
        print(f"识别实体: {entities}")

    except Exception as e:
        print(f"加载模型时出错: {e}")


# 运行测试
test_loaded_model()

print("训练和测试完成！")