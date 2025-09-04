import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

csv = pd.read_csv("dataset.csv", sep="\t", header=None)

jieba.add_word("墓王之王")
jieba.add_word("突变团竞")
jieba.add_word("一千六百五十三")

# 中文先分词，用空格隔开
lcut = csv[0].apply(lambda c1 : " ".join(jieba.lcut(c1)))

# 创建向量化器并拟合
vector = CountVectorizer()
# 建立词汇表
vector.fit(lcut)
# 真正转换成向量(稀疏矩阵)
X = vector.transform(lcut)

X_train, X_test, y_train, y_test = train_test_split(
    X, csv[1], test_size=0.2, random_state=42, stratify=csv[1]
)

# 测试
test_input = "周末去哪玩"
test_lcut = " ".join(jieba.lcut(test_input))
test_vector = vector.transform([test_lcut])

# 朴素贝叶斯
model_nb = MultinomialNB()
model_nb.fit(X_train, y_train)
print("MultinomialNB train score={}".format(model_nb.score(X_train, y_train)))
print("MultinomialNB test score={}".format(model_nb.score(X_test, y_test)))
print(model_nb.predict(test_vector))

# 逻辑回归
model_logis = LogisticRegression()
model_logis.fit(X_train, y_train)
print("LogisticRegression train score={}".format(model_logis.score(X_train, y_train)))
print("LogisticRegression test score={}".format(model_logis.score(X_test, y_test)))
print(model_logis.predict(test_vector))

# 获取类别和概率
# classes = model_logis.classes_
# probs = model_logis.predict_proba(test_vector)[0]
# 制表
# df_prob = pd.DataFrame({
#     "类别": classes,
#     "预测概率": probs
# })
# df_prob = df_prob.sort_values(by="预测概率", ascending=False).reset_index(drop=True)
# print(df_prob)

y_pred = model_logis.predict(X_test)
print(classification_report(y_test, y_pred))
