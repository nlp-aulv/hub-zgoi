import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_Sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vectorizer = CountVectorizer()
input_feature = vectorizer.fit_transform(input_Sentence.values)

test_query = ["我要出去旅游几天，帮我把灯设置为离家模式吧", "播放一首好日子"]
test_sentence = [" ".join(jieba.cut(text)) for text in test_query]
print(test_sentence)
test_feature = vectorizer.transform(test_sentence)
print("待预测文本：", test_query)

# KNN算法
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1])
print("KNN模型预测结果：", model.predict(test_feature))

# 线性回归
model = LogisticRegression(max_iter=1000)
model.fit(input_feature, dataset[1])
print("线性回归预测结果：", model.predict(test_feature))

# 决策树
model = DecisionTreeClassifier(max_depth=5)
model.fit(input_feature, dataset[1])
print("决策树预测结果：", model.predict(test_feature))

# 线性SVC
model = LinearSVC(max_iter=10000)
model.fit(input_feature, dataset[1])
print("线性SVC预测结果：", model.predict(test_feature))

# 决策森林
model = RandomForestClassifier(n_estimators=100, max_depth=5)
model.fit(input_feature, dataset[1])
print("决策森林预测结果：", model.predict(test_feature))
