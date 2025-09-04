import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

#数据处理
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

x_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
y_target = dataset[1].values

test_text = "帮我播放一下郭德纲的小品"
test_x = " ".join(jieba.lcut(test_text))

#贝叶斯模型
tfidf_vector = TfidfVectorizer()
x_feature = tfidf_vector.fit_transform(x_sentence)
bs_model = MultinomialNB()
bs_model.fit(x_feature, y_target)

#贝叶斯验证
test_feature = tfidf_vector.transform([test_x])
print("待预测的文本", test_text)
print("贝叶斯模型预测结果: ", bs_model.predict(test_feature))

#SVM模型
count_vector = CountVectorizer()
x_feature = count_vector.fit_transform(x_sentence)
svm_model = SVC(kernel="linear", C=1.0, class_weight="balanced")
svm_model.fit(x_feature, y_target)

#SVM验证
test_feature = count_vector.transform([test_x])
print("待预测的文本", test_text)
print("svm模型预测结果: ", svm_model.predict(test_feature))