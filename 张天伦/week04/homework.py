import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset


class BertTextClassifier:
    """BERT文本分类器"""

    def __init__(self, model_path='../../models/google-bert/bert-base-chinese'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_and_preprocess_data(self, file_path='./waimai_10k.csv'):
        """加载和预处理数据"""
        data_frame = pd.read_csv(file_path)

        # 提取文本和标签
        text_sequences = data_frame['review'].values.tolist()
        target_labels = data_frame['label'].values.tolist()

        return text_sequences, target_labels

    def create_data_splits(self, texts, labels, test_fraction=0.2, sample_ratio=0.2, random_seed=42):
        """创建训练测试分割并进行采样"""
        # 初始分割
        train_texts, test_texts, train_lbls, test_lbls = train_test_split(
            texts, labels,
            test_size=test_fraction,
            stratify=labels,
            random_state=random_seed
        )

        # 对训练集进行采样
        train_sample_count = int(len(train_texts) * sample_ratio)
        train_selected_indices = np.random.choice(
            len(train_texts), train_sample_count, replace=False
        )
        sampled_train_texts = [train_texts[idx] for idx in train_selected_indices]
        sampled_train_labels = [train_lbls[idx] for idx in train_selected_indices]

        # 对测试集进行采样
        test_sample_count = int(len(test_texts) * sample_ratio)
        test_selected_indices = np.random.choice(
            len(test_texts), test_sample_count, replace=False
        )
        sampled_test_texts = [test_texts[idx] for idx in test_selected_indices]
        sampled_test_labels = [test_lbls[idx] for idx in test_selected_indices]

        return sampled_train_texts, sampled_test_texts, sampled_train_labels, sampled_test_labels

    def initialize_model_and_tokenizer(self, num_categories=12):
        """初始化BERT模型和分词器"""
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=num_categories
        )

    def tokenize_texts(self, text_list, max_seq_length=64):
        """对文本进行分词处理"""
        return self.tokenizer(
            text_list,
            truncation=True,
            padding=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )

    def create_dataset(self, encodings, labels):
        """创建HF数据集"""
        return Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        })

    @staticmethod
    def calculate_metrics(evaluation_prediction):
        """计算评估指标"""
        model_logits, true_labels = evaluation_prediction
        predicted_classes = np.argmax(model_logits, axis=-1)
        accuracy_score = np.mean(predicted_classes == true_labels)
        return {'accuracy': accuracy_score}

    def setup_training_parameters(self):
        """设置训练参数"""
        return TrainingArguments(
            output_dir='./assets/weights/bert/',
            num_train_epochs=4,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            logging_steps=100,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
        )

    def execute_training(self, train_data, eval_data, training_params):
        """执行训练过程"""
        model_trainer = Trainer(
            model=self.model,
            args=training_params,
            train_dataset=train_data,
            eval_dataset=eval_data,
            compute_metrics=self.calculate_metrics,
        )

        # 训练模型
        model_trainer.train()

        # 评估模型
        evaluation_results = model_trainer.evaluate()
        print(f"评估结果: {evaluation_results}")

        return model_trainer

    def preserve_best_model(self, trainer_instance, save_path='./weights/bert.pt'):
        """保存最佳模型"""
        best_model_checkpoint = trainer_instance.state.best_model_checkpoint

        if best_model_checkpoint:
            optimal_model = BertForSequenceClassification.from_pretrained(best_model_checkpoint)
            torch.save(optimal_model.state_dict(), save_path)
            print(f"最佳模型已保存至: {save_path}")
            return optimal_model
        else:
            print("未找到最佳模型检查点")
            return None


def main():
    """主执行函数"""
    # 初始化分类器
    text_classifier = BertTextClassifier()

    # 加载数据
    text_data, label_data = text_classifier.load_and_preprocess_data()

    # 创建数据分割
    (train_texts, test_texts,
     train_labels, test_labels) = text_classifier.create_data_splits(text_data, label_data)

    # 初始化模型
    text_classifier.initialize_model_and_tokenizer(num_categories=12)

    # 分词处理
    train_tokenized = text_classifier.tokenize_texts(train_texts)
    test_tokenized = text_classifier.tokenize_texts(test_texts)

    # 创建数据集
    training_dataset = text_classifier.create_dataset(train_tokenized, train_labels)
    testing_dataset = text_classifier.create_dataset(test_tokenized, test_labels)

    # 设置训练参数
    training_config = text_classifier.setup_training_parameters()

    # 执行训练
    trainer = text_classifier.execute_training(training_dataset, testing_dataset, training_config)

    # 保存最佳模型
    text_classifier.preserve_best_model(trainer)


if __name__ == "__main__":
    main()
