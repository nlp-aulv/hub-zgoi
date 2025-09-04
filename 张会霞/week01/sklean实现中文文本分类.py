import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


# 读取训练集数据，两列dataset[0], dataset[1]
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset.head(5))
# print(len(set(dataset[1])))         # 文本分类的个数
# print(set(dataset[1]))              # 文本分类有哪些
# print(type(dataset[0]))            # <class 'pandas.core.series.Series'>

# 提取 文本的特征 tfidf， dataset[0]
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
# print(model)

# 构建一个逻辑回归模型进行训练
model_logic = LogisticRegression(max_iter=1000)
model_logic.fit(input_feature, dataset[1].values)

# 构建一个朴素贝叶斯算法进行训练
model_bayes = MultinomialNB()
model_bayes.fit(input_feature, dataset[1].values)

# 预测，用户输入的一个文本，进行预测结果
test_query = "找歌曲离别开发花并播放"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model.predict(test_feature))
print("LogisticRegression模型预测结果: ", model_logic.predict(test_feature))
print("NaiveBayes模型预测结果: ", model_bayes.predict(test_feature))

