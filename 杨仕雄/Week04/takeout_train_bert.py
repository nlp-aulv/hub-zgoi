import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import numpy as np

# 加载和预处理数据 dataset_df：DataFrame
dataset_df = pd.read_csv('E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\dataset\\takeout_comment_10k.csv', sep=',', header=0)

# 初始化 LabelEncoder，用于将文本标签转换为数字标签
lbl = LabelEncoder()
# 标签转换为数字标签
# labels = lbl.fit_transform(dataset_df['label'].values[:500]) # .values:转换为numpy数组  fit_transform()：fit → 学习所有类别，并为每个类别分配一个数字（内部保存映射关系）。transform → 把原始标签映射成对应的数字编码。
labels = lbl.fit_transform(dataset_df['label'].values[3752:4252])
# 提取差评好评都有的文本内容
texts = list(dataset_df['review'].values[3752:4252])


# 分割数据为训练集和测试集
x_train, x_test, train_labels, test_labels = train_test_split(
    texts,             # 文本数据
    labels,            # 对应的数字标签
    test_size=0.2,     # 测试集比例为20%
    stratify=labels    # 确保训练集和测试集的标签分布一致
)

# 从预训练模型加载分词器和模型
# 载分词器，把句子转成 BERT 输入格式（数字化的 tokens）
tokenizer = BertTokenizer.from_pretrained('E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\models\\bert-base-chinese\\')
# 加载带分类头的 BERT 模型，适用于分类任务（这里是 12 类）
model = BertForSequenceClassification.from_pretrained(
    'E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\models\\bert-base-chinese\\', num_labels=2 # 判别式模型，分多少类
)

# 使用分词器对训练集和测试集的文本进行编码
# truncation=True：如果文本过长则截断
# padding=True：对齐所有序列长度，填充到最长
# max_length=64：最大序列长度
train_encodings = tokenizer(x_train, truncation=True, padding=True, max_length=64)
test_encodings = tokenizer(x_test, truncation=True, padding=True, max_length=64)


# 将编码后的数据和标签转换为 Hugging Face `datasets` 库的 Dataset 对象
train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],           # 文本的token ID
    'attention_mask': train_encodings['attention_mask'], # 注意力掩码
    'labels': train_labels                               # 对应的标签
})
test_dataset = Dataset.from_dict({
    'input_ids': test_encodings['input_ids'],
    'attention_mask': test_encodings['attention_mask'],
    'labels': test_labels
})

# 定义用于计算评估指标的函数
def compute_metrics(eval_pred):
    # eval_pred 是一个元组，包含模型预测的 logits 和真实的标签
    logits, labels = eval_pred
    # 找到 logits 中最大值的索引，即预测的类别
    predictions = np.argmax(logits, axis=-1)
    # 计算预测准确率并返回一个字典
    return {'accuracy': (predictions == labels).mean()}

# 配置训练参数
training_args = TrainingArguments(
    output_dir='E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\weights\\bert\\', # 训练输出目录，用于保存模型和状态
    num_train_epochs=4,                  # 训练的总轮数
    per_device_train_batch_size=16,      # 训练时每个设备（GPU/CPU）的批次大小
    per_device_eval_batch_size=16,       # 评估时每个设备的批次大小
    warmup_steps=500,                    # 学习率预热的步数，有助于稳定训练
    weight_decay=0.01,                   # 权重衰减，用于防止过拟合
    logging_dir='./logs',                # 日志存储目录
    logging_steps=100,                   # 每隔100步记录一次日志
    eval_strategy="epoch",               # 每训练完一个 epoch 进行一次评估
    save_strategy="epoch",               # 每训练完一个 epoch 保存一次模型
    load_best_model_at_end=True,         # 训练结束后加载效果最好的模型
)

# 实例化 Trainer
trainer = Trainer(
    model=model,                         # 要训练的模型
    args=training_args,                  # 训练参数
    train_dataset=train_dataset,         # 训练数据集
    eval_dataset=test_dataset,           # 评估数据集
    compute_metrics=compute_metrics,     # 用于计算评估指标的函数
)

# 开始训练模型
trainer.train()
# 在测试集上进行最终评估
trainer.evaluate()

best_model_path = trainer.state.best_model_checkpoint
if best_model_path:
    best_model = BertForSequenceClassification.from_pretrained(best_model_path)
    print(f"The best model is located at: {best_model_path}")
    torch.save(best_model.state_dict(), 'E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\weights\\bert_takeout.pt')
    print("Best model saved to E:\\八斗学院资料\\project1\\01-intent-classify\\assets\\weights\\bert_takeout.pt")
else:
    print("Could not find the best model checkpoint.")
