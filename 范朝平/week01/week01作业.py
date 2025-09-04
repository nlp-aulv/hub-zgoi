import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn import linear_model, tree, neighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

# 加载数据集
data = pd.read_csv('dataset.csv', sep="\t", header=None)
print("数据集前5行:")
print(data.head(5))

# 数据预处理 将文本转换为特征向量
vectorizer = CountVectorizer()
input_sententce = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
X = vectorizer.fit_transform(input_sententce.values)
y = data[1].values

# 划分训练集和测试集
train_x, test_x, train_y, test_y = train_test_split(X, y, random_state=520, test_size=0.25)
print("训练集大小:", train_x.shape)
print("测试集大小:", test_x.shape)

# 逻辑回归
print("")
print("=== 逻辑回归 ===")
logistic_model = linear_model.LogisticRegression(max_iter=1000)
logistic_model.fit(train_x, train_y)
logistic_pred = logistic_model.predict(test_x)
logistic_acc = accuracy_score(test_y, logistic_pred)
print("准确率:", logistic_acc)

# 测试 逻辑回归模型
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vectorizer.transform([test_sentence])
print("待预测的文本", test_query)
print("逻辑回归模型预测结果: ", logistic_model.predict(test_feature))


# 决策树
print("")
print("=== 决策树 ===")
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(train_x, train_y)
tree_pred = tree_model.predict(test_x)
tree_acc = accuracy_score(test_y, tree_pred)
print("准确率:", tree_acc)

test_query2 = "帮我播放一下周星驰的电影"
test_sentence2 = " ".join(jieba.lcut(test_query2))
test_feature2 = vectorizer.transform([test_sentence2])
print("待预测的文本", test_query2)
print("决策树模型预测结果: ", logistic_model.predict(test_feature2))

# KNN
print("")
print("=== KNN ===")
for k in [1, 3, 5, 7, 9]:
    knn_model = neighbors.KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(train_x, train_y)
    knn_pred = knn_model.predict(test_x)
    knn_acc = accuracy_score(test_y, knn_pred)
    print(f"K={k} 准确率:", knn_acc)

test_query3 = "想去三亚玩两天"
test_sentence3 = " ".join(jieba.lcut(test_query3))
test_feature3 = vectorizer.transform([test_sentence3])
print("待预测的文本", test_query3)
print("KNN模型预测结果: ", logistic_model.predict(test_feature3))

