import jieba
import pandas
from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier

# \t 列分隔符
dataset = pandas.read_csv("dataset.csv", sep="\t", header=None)
# print(dataset)
input_sentences = dataset[0].apply(lambda input: " ".join(jieba.lcut(input)))
# 分词结果 
print(input_sentences)

vector = CountVectorizer()
vector.fit(input_sentences.values)
input_feature = vector.transform(input_sentences.values)
# print(input_feature)

# KNN
# 样本有12100个 扩大近邻元素的个数。。不然根本分不准
# 3579个位数和十位数数量级的邻居配置基本没有准的，101，201，301，401后面两个准了
#
k_neighbors_3 = KNeighborsClassifier(n_neighbors=1501)
k_neighbors_3.fit(input_feature, dataset[1].values)

k_neighbors_5 = KNeighborsClassifier(n_neighbors=2501)
k_neighbors_5.fit(input_feature, dataset[1].values)
# print(model)

k_neighbors_7 = KNeighborsClassifier(n_neighbors=3501)
k_neighbors_7.fit(input_feature, dataset[1].values)

k_neighbors_9 = KNeighborsClassifier(n_neighbors=4501)
k_neighbors_9.fit(input_feature, dataset[1].values)

# 逻辑回归
linear_model1 = linear_model.LogisticRegression(max_iter=1000)
linear_model1.fit(input_feature, dataset[1].values)
print(linear_model1)

linear_model2 = linear_model.LogisticRegression(max_iter=2000)
linear_model2.fit(input_feature, dataset[1].values)
print(linear_model2)

# 朴素贝叶斯
naive_bayes1 = MultinomialNB()
naive_bayes1.fit(input_feature, dataset[1].values)

naive_bayes2 = MultinomialNB(alpha=200)
naive_bayes2.fit(input_feature, dataset[1].values)

# 这个好像怎么都不准
test_query = "除了天天向上以外还有什么比较好看的综艺节目？"
test_sentence = " ".join(jieba.lcut(test_query))
test_feature = vector.transform([test_sentence])
print("待预测的文本", test_query)
print("KNN模型预测结果: ", k_neighbors_3.predict(test_feature), k_neighbors_5.predict(test_feature),k_neighbors_7.predict(test_feature),k_neighbors_9.predict(test_feature))
print("逻辑回归：", linear_model1.predict(test_feature), linear_model2.predict(test_feature))
print("朴素贝叶斯：", naive_bayes1.predict(test_feature), naive_bayes2.predict(test_feature))

# 也不准
test_query2 = "给我推荐点好看的电视剧"
test_sentence2 = " ".join(jieba.lcut(test_query2))
test_feature2 = vector.transform([test_sentence2])
print("待预测的文本2", test_query2)
print("KNN模型预测结果2: ", k_neighbors_3.predict(test_feature2), k_neighbors_5.predict(test_feature2),k_neighbors_7.predict(test_feature2),k_neighbors_9.predict(test_feature2))
print("逻辑回归：", linear_model1.predict(test_feature2), linear_model2.predict(test_feature2))
print("朴素贝叶斯：", naive_bayes1.predict(test_feature2), naive_bayes2.predict(test_feature2))

test_query3 = "请给我播放我喜爱的音乐，就现在，前两天刚听过的"
test_sentence3 = " ".join(jieba.lcut(test_query3))
test_feature3 = vector.transform([test_sentence3])
print("待预测的文本3", test_query3)
print("KNN模型预测结果3: ", k_neighbors_3.predict(test_feature3), k_neighbors_5.predict(test_feature3),k_neighbors_7.predict(test_feature3),k_neighbors_9.predict(test_feature3))
print("逻辑回归：", linear_model1.predict(test_feature3), linear_model2.predict(test_feature3))
print("朴素贝叶斯：", naive_bayes1.predict(test_feature3), naive_bayes2.predict(test_feature3))

test_query4 = "帮我看看明天下不下雨"
test_sentence4 = " ".join(jieba.lcut(test_query4))
test_feature4 = vector.transform([test_sentence4])
print("待预测的文本4", test_query4)
print("KNN模型预测结果4: ", k_neighbors_3.predict(test_feature4), k_neighbors_5.predict(test_feature4),k_neighbors_7.predict(test_feature4),k_neighbors_9.predict(test_feature4))
print("逻辑回归：", linear_model1.predict(test_feature4), linear_model2.predict(test_feature4))
print("朴素贝叶斯：", naive_bayes1.predict(test_feature4), naive_bayes2.predict(test_feature4))

# 样本集分割
train_x, test_x, train_y, test_y = train_test_split(dataset[0], dataset[1], test_size=0.1, random_state=520)
# print("train_x, train_y:", train_x, train_y)
# print("test_x, test_y", test_x, test_y)

input_train_x = train_x.apply(lambda test_input: " ".join(jieba.lcut(test_input)))
vector.fit(input_train_x.values)
input_feature_train_split = vector.transform(input_train_x.values)

# KNN
k_neighbors_3 = KNeighborsClassifier(n_neighbors=101)
k_neighbors_3.fit(input_feature_train_split, train_y.values)

k_neighbors_5 = KNeighborsClassifier(n_neighbors=201)
k_neighbors_5.fit(input_feature_train_split, train_y.values)
# print(model)

k_neighbors_7 = KNeighborsClassifier(n_neighbors=301)
k_neighbors_7.fit(input_feature_train_split, train_y.values)

k_neighbors_9 = KNeighborsClassifier(n_neighbors=1001)
k_neighbors_9.fit(input_feature_train_split, train_y.values)

# 逻辑回归
linear_model1 = linear_model.LogisticRegression(max_iter=1000)
linear_model1.fit(input_feature_train_split, train_y.values)
print(linear_model1)

linear_model2 = linear_model.LogisticRegression(max_iter=2000)
linear_model2.fit(input_feature_train_split, train_y.values)
print(linear_model2)

# 朴素贝叶斯
naive_bayes1 = MultinomialNB()
naive_bayes1.fit(input_feature_train_split, train_y.values)

naive_bayes2 = MultinomialNB(alpha=200)
naive_bayes2.fit(input_feature_train_split, train_y.values)

test_sentence_split = test_x.apply(lambda test_input: " ".join(jieba.lcut(test_input)))
test_feature_split = vector.transform(test_sentence_split)
print("待预测的文本：", test_x)
print("预期结果：", test_y)
print("KNN模型预测结果2: ", k_neighbors_3.predict(test_feature_split), k_neighbors_5.predict(test_feature_split),k_neighbors_7.predict(test_feature_split),k_neighbors_9.predict(test_feature_split))
print("逻辑回归：", linear_model1.predict(test_feature_split), linear_model2.predict(test_feature_split))
print("朴素贝叶斯：", naive_bayes1.predict(test_feature_split), naive_bayes2.predict(test_feature_split))


