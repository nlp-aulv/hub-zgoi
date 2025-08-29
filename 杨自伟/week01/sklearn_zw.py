import pandas as pd
import jieba
from sklearn.feature_extraction import text
from sklearn import tree
from sklearn import neighbors

#数据集
datas = pd.read_csv('dataset.csv',sep='\t',header=None)
# print(type(datas))

# 中文分词
data_zw = datas[0].apply(lambda x:' '.join(jieba.lcut(x)))
# print(type(data_zw.values),'\n',data_zw)
target = datas[1].values
# print(type(target),'\n',datas[1])

#提取特征向量化
vector = text.CountVectorizer()
vector.fit(data_zw.values)
data_vt = vector.transform(data_zw.values)

# 创建模型-决策树
model_tree = tree.DecisionTreeClassifier()
model_tree.fit(data_vt,target)

input_w = "看天气预报"
input_lc = ' '.join(jieba.lcut(input_w))
input_vt = vector.transform([input_lc])
prediction1 = model_tree.predict(input_vt)
print("查看内容：",input_w)
print("tree-预测结果：",prediction1)

# 创建模型-k-means
model_km = neighbors.KNeighborsClassifier(1)
model_km.fit(data_vt,target)

input_w = "我要听歌"
input_lc = ' '.join(jieba.lcut(input_w))
input_vt = vector.transform([input_lc])
prediction1 = model_km.predict(input_vt)
print("查看内容：",input_w)
print("kmeans-预测结果：",prediction1)
