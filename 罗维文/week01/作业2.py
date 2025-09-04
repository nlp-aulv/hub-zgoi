import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer  # 文本特征提取
from sklearn.model_selection import train_test_split  # 数据集划分
from sklearn.linear_model import LogisticRegression  # 逻辑回归
from sklearn.tree import DecisionTreeClassifier  # 决策树
from sklearn.neighbors import KNeighborsClassifier  # KNN模块

# 导入数据并处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 对文本进行特征提取
vector = CountVectorizer()
vector.fit(input_sentence.values)
intput_feature = vector.transform(input_sentence.values)

# 数据切分 10% 样本划分为测试集
train_x, test_x, train_y, test_y = train_test_split(intput_feature, dataset[1], test_size=0.1, stratify=dataset[1])

# 训练模型
Logistic_model = LogisticRegression()  # 逻辑回归
Logistic_model.fit(train_x, train_y)
Lgc_prediction = Logistic_model.predict(test_x)

Tree_model = DecisionTreeClassifier()  # 决策树
Tree_model.fit(train_x, train_y)
Tree_prediction = Tree_model.predict(test_x)

KNN_model = KNeighborsClassifier(n_neighbors=5)  # KNN
KNN_model.fit(train_x, train_y)
KNN_prediction = KNN_model.predict(test_x)

# 结果检测
print("总预测数：", len(test_y))
print("逻辑回归预测结果相同数:", (test_y == Lgc_prediction).sum())
print("决策树预测结果相同数:", (test_y == Tree_prediction).sum())
print("5-KNN预测结果相同数:", (test_y == KNN_prediction).sum())
