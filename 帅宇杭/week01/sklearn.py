import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn 对中文处理

vector = CountVectorizer() 
vector.fit(input_sentence.values)
input_feature = vector.transform(input_sentence.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)
print(model)

test_query = "播放一首张学友的歌"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果：", model.predict(test_feature))

model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
print(model)
test_query = "播放一首张学友的歌"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("决策树模型预测结果：", model.predict(test_feature))

model = LogisticRegression(max_iter=1000)
model.fit(input_feature, dataset[1].values)
print(model)
test_query = "播放一首张学友的歌"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("逻辑回归模型预测结果：", model.predict(test_feature))
