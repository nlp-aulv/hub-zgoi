import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

# 作业描述：读取dataset.csv文件，对文本进行分词，并使用两种分类算法进行分类。

# 数据加载
data = pd.read_csv('dataset.csv', sep="\t", header=None)
# 数据预处理
input_sententce = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
# 特征提取
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(input_sententce)  # 输入文本向量化
# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, data[1])  # 训练模型
# 逻辑回归
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X, data[1])  # 训练模型
# 预测新数据
test_query = "我想看刘德华的电影"
test_sentence = " ".join(jieba.lcut(test_query))
new_X = vectorizer.transform([test_sentence])  # 输入文本向量化
result_knn = knn.predict(new_X)  # 预测结果
print('KNN分类结果: ', result_knn)  # 输出分类结果
result_log_reg = log_reg.predict(new_X)  # 预测结果
print('逻辑回归分类结果: ', result_log_reg)  # 输出分类结果

# 以上代码输出结果：
# KNN分类结果:  ['Music-Play']
# 逻辑回归分类结果:  ['FilmTele-Play']


