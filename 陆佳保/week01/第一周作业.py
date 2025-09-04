import pandas as pd
import jieba
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings('ignore')

# 数据加载
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 文本预处理
input_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))

# 特征提取
vectorizer = TfidfVectorizer()
input_feature = vectorizer.fit_transform(input_sentence.values)
labels = dataset[1].values

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    input_feature, labels, test_size=0.2, random_state=42
)

# 测试文本
test_query = "帮我播放一下郭德纲的小品"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vectorizer.transform([test_sentence])

# 1. KNN模型
print("=== 1. KNN模型 ===")
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)
print(f"KNN准确率: {knn_accuracy:.4f}")
knn_test_pred = knn_model.predict(test_feature)
print(f"测试预测结果: {knn_test_pred[0]}")
print()

# 2. 朴素贝叶斯模型
print("=== 2. 朴素贝叶斯模型 ===")
from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
nb_predictions = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)
print(f"朴素贝叶斯准确率: {nb_accuracy:.4f}")
nb_test_pred = nb_model.predict(test_feature)
print(f"测试预测结果: {nb_test_pred[0]}")
print()

# 3. 支持向量机模型
print("=== 3. 支持向量机模型 ===")
from sklearn.svm import SVC

svm_model = SVC(kernel='linear', random_state=42)
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
print(f"SVM准确率: {svm_accuracy:.4f}")
svm_test_pred = svm_model.predict(test_feature)
print(f"测试预测结果: {svm_test_pred[0]}")
print()

# 4. 逻辑回归模型
print("=== 4. 逻辑回归模型 ===")
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_predictions)
print(f"逻辑回归准确率: {lr_accuracy:.4f}")
lr_test_pred = lr_model.predict(test_feature)
print(f"测试预测结果: {lr_test_pred[0]}")
print()

# 5. 随机森林模型
print("=== 5. 随机森林模型 ===")
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_predictions)
print(f"随机森林准确率: {rf_accuracy:.4f}")
rf_test_pred = rf_model.predict(test_feature)
print(f"测试预测结果: {rf_test_pred[0]}")
print()

# 6. 决策树模型
print("=== 6. 决策树模型 ===")
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)
print(f"决策树准确率: {dt_accuracy:.4f}")
dt_test_pred = dt_model.predict(test_feature)
print(f"测试预测结果: {dt_test_pred[0]}")
print()
