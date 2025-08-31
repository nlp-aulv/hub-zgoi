import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


#读取cxv文件
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

#提取文本特征
#学习词汇表
vector = CountVectorizer()
vector.fit(input_sentence.values)
#转换数据，输出稀疏矩阵
input_feature = vector.transform(input_sentence.values)

#KNN模型
modelKNN = KNeighborsClassifier()
modelKNN.fit(input_feature, dataset[1].values)

#逻辑回归模型
modelLogic = LogisticRegression()
modelLogic.fit(input_feature, dataset[1].values)

#随机森林模型
modelForest = RandomForestClassifier()
modelForest.fit(input_feature, dataset[1].values)

#神经网络模型
modelMLP = MLPClassifier(hidden_layer_sizes=(100,))
modelMLP.fit(input_feature, dataset[1].values)


test_query = "我想去看哈利波特大电影"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待遇测文本:" + test_query)
print("KNN预测结果：" + modelKNN.predict(test_feature))
print("逻辑回归预测结果：" + modelLogic.predict(test_feature))
print("随机森林模型：" + modelForest.predict(test_feature))
prediction = modelMLP.predict(test_feature)[0]
print("神经网络模型：" + prediction)
print("神经网络模型：{}".format(modelMLP.predict(test_feature)[0]))
