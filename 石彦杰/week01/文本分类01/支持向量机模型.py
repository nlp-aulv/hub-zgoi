# 文本分类 - 朴素贝叶斯模型
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# 设置图形样式
plt.style.use('ggplot')

print("加载数据集...")

# 使用制表符作为分隔符，并指定列名
df = pd.read_csv('dataset.csv', sep='\t', header=None, names=['text', 'label'])

print("数据集基本信息:")
print(f"数据集形状: {df.shape}")
print(f"\n前5行数据:")
print(df.head())
print(f"\n列名: {df.columns.tolist()}")
print(f"\n标签分布:\n{df['label'].value_counts()}")

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 如果有缺失值，删除含有缺失值的行
df = df.dropna()

print("\n数据预处理和特征提取...")
# 文本预处理和TF-IDF向量化
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)

X = tfidf_vectorizer.fit_transform(df['text'])
y = df['label']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")

print("\n训练朴素贝叶斯模型...")
# 训练朴素贝叶斯模型
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# 预测
nb_pred = nb_model.predict(X_test)

# 评估
nb_accuracy = accuracy_score(y_test, nb_pred)
print("朴素贝叶斯模型准确率: {:.2f}%".format(nb_accuracy * 100))
print("\n朴素贝叶斯分类报告:")
print(classification_report(y_test, nb_pred))

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, nb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('朴素贝叶斯混淆矩阵')
plt.ylabel('真实标签')
plt.xlabel('预测标签')
plt.tight_layout()
plt.savefig('nb_confusion_matrix.png')
plt.show()

# 分析关键特征
feature_names = tfidf_vectorizer.get_feature_names_out()
for i, class_name in enumerate(nb_model.classes_):
    # 获取每个类别的对数概率
    log_probs = nb_model.feature_log_prob_[i]
    # 获取前10个最重要的特征
    top10 = np.argsort(log_probs)[-10:]
    top10_features = [feature_names[j] for j in top10]
    print(f"类别 '{class_name}' 的最重要特征: {top10_features}")

print("\n朴素贝叶斯模型训练和评估完成!")
