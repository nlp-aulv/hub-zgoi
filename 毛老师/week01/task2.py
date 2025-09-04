import jieba
import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. 数据加载与预处理 ---
print("--- 1. 正在加载和预处理数据 ---")
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None, nrows=100)

print("数据集前5行预览：")
print(dataset.head(5))

# 使用jieba进行中文分词，并用空格分隔
# sklearn的CountVectorizer默认使用空格作为分隔符
input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
labels = dataset[1]

# --- 2. 特征提取 (词袋模型) ---
print("\n--- 2. 正在进行特征提取 ---")
vectorizer = CountVectorizer()
input_features = vectorizer.fit_transform(input_sentences.values)
print(f"特征矩阵的形状： {input_features.shape}")

# --- 3. 模型训练与评估 ---
# 不再需要手动划分训练集和测试集，cross_val_predict 会处理交叉验证的划分

# 定义四种不同的模型
models = {
    'KNN (K-Nearest Neighbors)': KNeighborsClassifier(),
    'Naive Bayes (MultinomialNB)': MultinomialNB(),
    'SVM (Support Vector Classifier)': SVC(kernel='linear'),
    'Decision Tree': DecisionTreeClassifier()
}

for name, model in models.items():
    print(f"\n--- 正在评估模型: {name} ---")

    # 使用 cross_val_predict 进行5折交叉验证，获取每个样本的预测结果
    y_pred = cross_val_predict(model, input_features, labels, cv=5)

    # 基于交叉验证的预测结果，计算并打印准确率和分类报告
    print(f"平均准确率: {accuracy_score(labels, y_pred):.4f}")
    report = classification_report(labels, y_pred, zero_division=0)
    print(report)
