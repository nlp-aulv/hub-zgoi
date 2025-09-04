from sklearn import tree  # 决策树模块
from sklearn import linear_model  # 线性模型模块
from sklearn.neighbors import KNeighborsClassifier  # KNN模型
from sklearn.model_selection import train_test_split  # 数据集划分
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
X, y = dataset[0], dataset[1]

# 数据切分 25% 样本划分为测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=666)

# sklearn对中文的处理
input_sentence_train_x = train_x.apply(lambda x: " ".join(jieba.lcut(x)))  # sklearn对中文的处理
input_sentence_test_x = test_x.apply(lambda x: " ".join(jieba.lcut(x)))

vector_train_x = CountVectorizer()  # 对文本进行提取特征 默认使用标点符号分词
vector_train_x.fit(input_sentence_train_x.values)
input_feature_train_x = vector_train_x.transform(input_sentence_train_x.values)

vector_test_x = CountVectorizer()  # 对文本进行提取特征 默认使用标点符号分词
vector_test_x.fit(input_sentence_test_x.values)
input_feature_test_x = vector_test_x.transform(input_sentence_test_x.values)

# KNN模型
knn_model = KNeighborsClassifier()  # 模型初始化
knn_model.fit(input_feature_train_x, train_y.values)
print(knn_model)

# Tree模型
tree_model = tree.DecisionTreeClassifier()  # 模型初始化
tree_model.fit(input_feature_train_x, train_y.values)
print(tree_model)

# 线性模块
line_model = linear_model.LogisticRegression(max_iter=10000)  # 模型初始化
line_model.fit(input_feature_train_x, train_y.values)
print(line_model)

# 带预测文本
test_query = "大家今天过得好吗？"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector_train_x.transform([test_sentence])
print("带预测的文本：", test_query)

# KNN预测
print("KNN模型预测结果：", knn_model.predict(test_feature))
# Tree预测
print("Tree模型预测结果：", tree_model.predict(test_feature))
# 线性模块
print("模型预测结果：", line_model.predict(test_feature))
