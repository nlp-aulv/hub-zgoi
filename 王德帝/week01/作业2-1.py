import jieba
import pandas as pd
from sklearn import neighbors
from sklearn.feature_extraction.text import CountVectorizer
from typing import List
from sklearn.model_selection import train_test_split

#读取数据
data = pd.read_csv("./week01/dataset.csv",sep='\t', header=None)
#分词
input_sententce:List[str] = data[0].apply(lambda x : " ".join(jieba.lcut(x)))
#print(input_sententce.head(10))

#提取特征:将文本数据转换为数值特征向量
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)
#print(input_feature)

#划分训练集和测试集
train_x,test_x,train_y,test_y = train_test_split(input_feature,data[1],test_size=0.2)

#训练模型
model = neighbors.KNeighborsClassifier(n_neighbors=5)
model.fit(train_x, train_y)

print("真实结果：\n",test_y)
print("预测的结果：\n",model.predict(test_x))
print("预测准确率：",(test_y==model.predict(test_x)).sum()/len(test_y))
#0.715702479338843
#0.7161157024793389
#0.721900826446281
#0.7272727272727273
#0.7140495867768595

#测试模型
# test_query = "我要看哪吒"
# test_sentence = " ".join(jieba.lcut(test_query))
# test_feature = vector.transform([test_sentence])
# print("预测文本：",test_query)
# print("预测结果：",model.predict(test_feature))
