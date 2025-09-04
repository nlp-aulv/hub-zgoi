import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB  # 新增朴素贝叶斯模型
from sklearn.metrics import accuracy_score  # 新增评估指标

dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
print(dataset.head(5))

input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))  # 分词处理

vector = CountVectorizer()  # 特征提取
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

# 模型1: KNN
model_knn = KNeighborsClassifier()
model_knn.fit(input_feature, dataset[1].values)
print("KNN模型训练完成")

# 模型2: 朴素贝叶斯
model_nb = MultinomialNB()
model_nb.fit(input_feature, dataset[1].values)
print("朴素贝叶斯模型训练完成")

# 测试预测
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])

print("待预测的文本:", test_query)
print("KNN模型预测结果:", model_knn.predict(test_feature))
print("朴素贝叶斯模型预测结果:", model_nb.predict(test_feature))

# 评估模型性能（可选）
# 假设数据集已经划分好训练集和测试集，可以添加以下代码：
y_pred_knn = model_knn.predict(input_feature)
y_pred_nb = model_nb.predict(input_feature)
print("KNN准确率:", accuracy_score(dataset[1].values, y_pred_knn))
print("朴素贝叶斯准确率:", accuracy_score(dataset[1].values, y_pred_nb))



