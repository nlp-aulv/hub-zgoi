import pandas as pd
import jieba
from typing import List
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

#读取数据
data = pd.read_csv("./week01/dataset.csv",sep='\t', header=None)

#分词
input_sententce:List[str] = data[0].apply(lambda x : " ".join(jieba.lcut(x)))

#特征提取
vector = CountVectorizer()
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

#训练集和测试集划分
train_x,test_x,train_y,test_y = train_test_split(input_feature,data[1],test_size=0.2)

#训练模型
model = linear_model.LogisticRegression()
model.fit(train_x,train_y)

#预测
# print("真实结果：\n",test_y.head(10))
# print("预测的结果：\n",model.predict(test_x)[:10])
# print("前10个样本预测准确率：",(test_y.head(10)==model.predict(test_x)[:10]).sum()/10)

print("真实结果：\n",test_y)
print("预测的结果：\n",model.predict(test_x))
print("预测准确率：",(test_y==model.predict(test_x)).sum()/len(test_y))
#0.8995867768595042
#0.8900826446280992
#0.881404958677686
#0.903305785123967
#0.9004132231404959
