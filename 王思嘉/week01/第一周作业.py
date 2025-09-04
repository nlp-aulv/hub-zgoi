#本次作业使用了knn和决策树模型
import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree


dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理
vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

#构建knn分类模型
model1 = KNeighborsClassifier()
model1.fit(input_feature, dataset[1].values)
print(model1)


test_query = "帮我播放一下knights的歌曲"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", model1.predict(test_feature))

#构建决策树分类模型
model2 = tree.DecisionTreeClassifier()
model2.fit(input_feature, dataset[1].values)
print(model2)
print("决策树模型预测结果: ", model2.predict(test_feature))
