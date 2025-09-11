import jieba
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 新词发现算法： 按照字之间的联合出现的频次，发现新的成语
#  -> 特定的场景
# subword token


# 随机森林训练
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(input_feature, dataset[1].values)

test_query = "我要听频率是一零五点六的助眠电台"
print("待预测的文本", test_query)

test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
# 预测
rf_pred = rf_model.predict(test_feature)

print(rf_pred)
