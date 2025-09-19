import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据
dataset_df = pd.read_csv('assets/dataset/作业数据-waimai_10k.csv', sep=',', encoding='UTF-8', quotechar='"', skipinitialspace=True)

print("数据形状:", dataset_df.shape)
print("列名:", dataset_df.columns.tolist())

# 初始化 LabelEncoder
lbl = LabelEncoder()

# 使用列名访问
labels = lbl.fit_transform(dataset_df['label'].values[:500])
texts = dataset_df['review'].values[:500]

print(f"处理了 {len(texts)} 个样本")
print(f"标签类别: {lbl.classes_}")
print(f"实际类别数量: {len(lbl.classes_)}")

# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels,   # 确保训练集和测试集的标签分布一致
    random_state=42    # 添加随机种子确保可重复性
)

print(f"训练集: {len(x_train)} 样本")
print(f"测试集: {len(x_test)} 样本")

# 从预训练模型加载分词器和模型
tokenizer = BertTokenizer.from_pretrained('./assets/models/bert-base-chinese')

# 使用实际的类别数量
model = BertForSequenceClassification.from_pretrained(
    './assets/models/bert-base-chinese',
    num_labels=len(lbl.classes_)
)

# 确保输入是字符串列表
x_train_list = [str(text) for text in x_train]
x_test_list = [str(text) for text in x_test]


train_labels = torch.tensor(train_labels, dtype=torch.long).float()
test_labels = torch.tensor(test_labels, dtype=torch.long).float()

print(f"训练标签数据类型: {train_labels.dtype}")
print(f"测试标签数据类型: {test_labels.dtype}")




# 使用分词器对训练集和测试集的文本进行编码
train_encodings = tokenizer(
    x_train_list,      # 使用处理后的列表
    truncation=True,
    padding=True,
    max_length=64
)
test_encodings = tokenizer(
    x_test_list,       # 使用处理后的列表
    truncation=True,
    padding=True,
    max_length=64
)





# 将编码后的数据和标签转换为 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})


# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = (predictions == labels).mean()
    return {'accuracy': accuracy}

# 配置训练参数 - 使用旧版本的参数名
training_args = TrainingArguments(
    output_dir='./assets/weights/bert/',
    num_train_epochs=4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    eval_strategy="epoch",               # 旧版本使用 eval_strategy
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=2,
)

# 实例化 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 开始训练模型
print("开始训练...")
trainer.train()

# 在测试集上进行最终评估
print("开始评估...")
eval_results = trainer.evaluate()
print(f"评估结果: {eval_results}")

# 保存最佳模型
best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    print(f"最佳模型路径: {best_model_path}")
    # 直接使用trainer的保存方法
    trainer.save_model('./assets/weights/bert-best')
    print("最佳模型已保存到 assets/weights/bert-best/")
else:
    # 如果没有最佳检查点，保存最终模型
    trainer.save_model('./assets/weights/bert-final')
    print("最终模型已保存到 assets/weights/bert-final/")

print("训练完成！")
