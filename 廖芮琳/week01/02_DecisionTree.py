from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import HashingVectorizer

import jieba
import pandas as pd

dataset  = pd.read_csv("dataset.csv", sep='\t', header=None)
# print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

vector = HashingVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = DecisionTreeClassifier()
model.fit(input_feature, dataset[1].values)
# print(model)

user_input = input("请输入需要分类的文本:")
test_sentence = " ".join(jieba.lcut(user_input))
test_feature = vector.transform([test_sentence])
print("待预测的文本:", user_input)
print("决策树模型预测结果: ", model.predict(test_feature))

