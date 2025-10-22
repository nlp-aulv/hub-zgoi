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
import codecs

tag_type = ['O', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC']
id2label = {i: label for i, label in enumerate(tag_type)}
label2id = {label: i for i, label in id2label.items()}


# 数据加载和预处理
def load_and_preprocess_data():
    """加载和预处理数据"""
    train_lines = codecs.open('sentences.txt', encoding='utf8').readlines()[:1000]
    train_lines = [x.replace(' ', '').strip() for x in train_lines]
    # train_lines = [list(text) for text in train_lines]

    train_tags = codecs.open('tags.txt').readlines()[:1000]
    train_tags = [x.strip().split(' ') for x in train_tags]
    # train_tags = [[label2id[x] for x in tag] for tag in train_tags]

    train_data = []
    for text, tags in zip(train_lines, train_tags):
        # 将序列标注转换为实体提取的描述格式
        entities = extract_entities_from_tags(list(text), tags)
        instruction = f"请从以下文本中提取命名实体：{text}"

        # 格式化输出：实体类型: 实体内容
        output_parts = []
        for entity_type, entity_text in entities:
            output_parts.append(f"{entity_type}: {entity_text}")

        output = ";".join(output_parts) if output_parts else "未识别到实体"

        train_data.append({
            "instruction": instruction,
            "input": "",
            "output": output
        })

    # 创建DataFrame
    df = pd.DataFrame(train_data)
    ds = Dataset.from_pandas(df)
    print(f"训练数据数量: {len(ds)}")
    print("示例数据:")
    for i in range(min(2, len(ds))):
        print(f"指令: {ds[i]['instruction']}")
        print(f"输出: {ds[i]['output']}")
        print("---")

    return ds


def extract_entities_from_tags(tokens, tags):
    """从标签序列中提取实体"""
    entities = []
    current_entity = ""
    current_type = ""

    for token, tag in zip(tokens, tags):
        if tag.startswith('B-'):
            # 保存前一个实体
            if current_entity:
                entities.append((current_type, current_entity))
            # 开始新实体
            current_entity = token
            current_type = tag[2:]  # 去掉B-前缀
        elif tag.startswith('I-') and current_entity:
            # 继续当前实体
            entity_type = tag[2:]
            if entity_type == current_type:  # 确保类型一致
                current_entity += token
            else:
                # 类型不匹配，结束当前实体
                if current_entity:
                    entities.append((current_type, current_entity))
                current_entity = ""
                current_type = ""
        else:
            # 结束当前实体
            if current_entity:
                entities.append((current_type, current_entity))
            current_entity = ""
            current_type = ""

    # 处理最后一个实体
    if current_entity:
        entities.append((current_type, current_entity))

    return entities


# 初始化tokenizer和模型
def initialize_model_and_tokenizer(model_path):
    """初始化tokenizer和模型"""
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度减少内存占用
        trust_remote_code=True
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
    instruction_text = f"<|im_start|>system\n你是一个实体抽取专家，需要从文本中识别出人名(PER)、地点(LOC)、组织机构(ORG)实体。<|im_end|>\n<|im_start|>user\n" \
        f"{example['instruction'] + example['input']}<|im_end|>\n<|im_start|>assistant\n"
    instruction = tokenizer(instruction_text, add_special_tokens=False)

    # 构建响应部分
    response = tokenizer(f"{example['output']}<im_end>", add_special_tokens=False)

    # 组合输入ID和注意力掩码
    input_ids = instruction["input_ids"] + response["input_ids"]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]

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
        output_dir="./ner_output_Qwen1.5",
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


# 预测函数
def predict_ner(model, tokenizer, text, device='cpu'):
    """预测单个文本的命名实体"""
    messages = [
        {"role": "system", "content": "你是一个实体抽取专家，需要从文本中识别出人名(PER)、地点(LOC)、组织机构(ORG)实体。"},
        {"role": "user", "content": text}
    ]

    # 应用聊天模板
    formatted_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print('聊天模板创建成功')

    # Tokenize输入
    model_inputs = tokenizer([formatted_text], return_tensors="pt").to(device)

    print('生成预测')
    # 生成预测
    with torch.no_grad():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=128,
            do_sample=False,
            temperature=0.1,  # 降低温度以获得更确定的输出
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    print('开始提取生成文本')
    # 提取生成的文本（去掉输入部分）
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print('提取完成')

    entities = parse_ner_response(response)
    return entities, response

    return response.strip()


def parse_ner_response(response):
    """解析模型生成的NER响应"""
    entities = []
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


# 批量预测
def batch_predict_ner(model, tokenizer, test_texts, device='cuda'):
    """批量预测NER"""
    results = []

    for text in tqdm(test_texts, desc="预测命名实体"):
        try:
            entities, raw_response = predict_ner(model, tokenizer, text, device)
            results.append({
                "text": text,
                "entities": entities,
                "raw_response": raw_response
            })
        except Exception as e:
            print(f"预测文本 '{text}' 时出错: {e}")
            results.append({
                "text": text,
                "entities": [],
                "raw_response": ""
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
    model_path = "./Qwen3-0.6B"
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
            pad_to_multiple_of=8  # 优化GPU内存使用
        ),
    )

    trainer.train()

    # 8. 保存模型
    print("保存模型...")
    trainer.save_model()
    tokenizer.save_pretrained("./ner_output_Qwen")


# 单独测试函数
def test_ner_example():
    # 下载模型
    # modelscope download --model Qwen/Qwen3-0.6B  --local_dir Qwen/Qwen3-0.6B
    model_path = "./Qwen3-0.6B/"
    tokenizer, model = initialize_model_and_tokenizer(model_path)

    # 加载训练好的LoRA权重
    model.load_adapter("./ner_output_Qwen1.5/")
    # model.cpu()
    print('模型加载完成')

    # 测试预测
    test_text = ["张三去了北京清华大学。"]
    print('开始测试')
    results = batch_predict_ner(model, tokenizer, test_text)
    # print(results)
    for result in results:
        print(f"\n文本: {result['text']}")
        if result['entities']:
            print("识别到的实体:")
            for entity, entity_type in result['entities']:
                entity = entity.split('<im_end>')[0]
                print(f"  {entity_type}: {entity}")
        else:
            print("未识别到实体")
        print("原始响应:", result['raw_response'].split('<im_end>')[0])
        print("-" * 80)


if __name__ == "__main__":
    # 执行主函数
    result_df = main()

    # 单独测试
    test_ner_example()
