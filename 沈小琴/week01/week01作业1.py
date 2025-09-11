# 中文分词工具
import jieba
# 数据处理库
import pandas as pd
#  数值计算库
import numpy as np
# CountVectorizer: 将文本转换为词频矩阵
from sklearn.feature_extraction.text import CountVectorizer
# LogisticRegression和SVC: 分类模型
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# train_test_split: 数据集划分工具
from sklearn.model_selection import train_test_split


# 从dataset.csv读取数据，假设数据以制表符分隔，没有表头
dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
# 对第一列的每个句子进行分词，并用空格连接分词结果
input_sentence_list = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))
# print(input_sentence_list)

# 创建CountVectorizer对象(CountVectorizer: 将文本转换为词频矩阵)
vector = CountVectorizer()


# 自然语言处理（NLP）中一个典型的向量化操作步骤
# vector 通常指一个文本向量化对象（如 CountVectorizer或 TfidfVectorizer等）
# .fit() 是机器学习中通用的方法，用于根据输入数据学习特征（如构建词表、计算权重等）
# input_sentence_list.values 包含(上面获取的'dataset.csv')文本数据的可迭代对象（如列表、Pandas列等）
vector.fit(input_sentence_list.values)


# 文本向量化的关键步骤，它将原始文本转换为机器学习模型可处理的数值特征。
# 将 已训练好的文本向量化器（vector）应用到新文本数据上，生成 数值特征矩阵（通常是稀疏矩阵）
# vector向量化对象
# transform()使用预定义的词表和规则转换新文本
# input_sentence_list.values 包含(上面获取的'dataset.csv')文本数据的可迭代对象（如列表、Pandas列等）
input_features_list = vector.transform(input_sentence_list.values)

# 将数据集按8:2的比例划分为训练集和测试集
# random_state=1037确保每次划分结果一致
train_X, test_X, train_Y, test_Y = train_test_split(input_features_list, dataset[1].values, test_size=0.2, random_state=1037)

# 训练逻辑回归模型
model1 = LogisticRegression()
model1.fit(train_X, train_Y)
# 训练线性核的SVM模型
model2 = SVC(kernel='linear')
model2.fit(train_X, train_Y)

# test model
# 在测试集上评估两个模型的性能
# 输出预测正确的样本数和总测试样本数
predictions = model1.predict(test_X)
print('LogisticRegression results:', (np.array(test_Y) == np.array(predictions)).sum(), test_X.shape[0])
predictions = model2.predict(test_X)
print('SVM results:', (np.array(test_Y) == np.array(predictions)).sum(), test_X.shape[0])

#eval model
# 对新句子"请播放郭德纲小品"进行分类
# 先分词，然后转换为特征向量
eval_quary = '请播放郭德纲小品'
eval_sentence = ' '.join(jieba.lcut(eval_quary))
eval_feature = vector.transform([eval_sentence])
# 输出两个模型的预测结果
print('LogisticRegression prediction:', model1.predict(eval_feature))
print('SVM prediction:', model2.predict(eval_feature))
