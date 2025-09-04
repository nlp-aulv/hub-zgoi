import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import neighbors
from sklearn import linear_model
import jieba


# 读取文件
data = pd.read_csv('dataset.csv', sep='\t', header=None)

# 切分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(data[0], data[1], random_state=1024)
# print(test_x, test_y)

# 分词
train_x = [' '.join(jieba.lcut(x)) for x in train_x]
test_x = [' '.join(jieba.lcut(x)) for x in test_x]

# 文本转化为向量
vector = CountVectorizer()
vector.fit(train_x)
train_x = vector.transform(train_x)
test_x = vector.transform(test_x)

test_query = '杭州天气怎么样？'
print('待预测文本：', test_query)
test_query = ' '.join(jieba.lcut(test_query))
# 模型
lin_model = linear_model.LogisticRegression(max_iter=1000)
lin_model.fit(train_x, train_y)
pred = lin_model.predict(test_x)
# print('liner-model预测结果：', pred)
print('liner-model准确率：', round((test_y == pred).sum() / len(test_y), 4))
print('liner-model预测结果：', lin_model.predict(vector.transform([test_query])))
print('= = == == = == = =')

tree_model = tree.DecisionTreeClassifier()
tree_model.fit(train_x, train_y)
pred = tree_model.predict(test_x)
print('tree-model准确率：', round((test_y == pred).sum() / len(test_y), 4))
print('tree-model预测结果：', tree_model.predict(vector.transform([test_query])))
print('= = == == = == = =')

for k in [1, 3, 5, 7, 9]:
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_x, train_y)
    pred = knn_model.predict(test_x)
    print(f'{k}-knn-model准确率：', round((test_y == pred).sum() / len(test_y), 4))
    print(f'{k}-knn-model预测结果：', knn_model.predict(vector.transform([test_query])))
    print('= = == == = == = =')
