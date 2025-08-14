import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# read data and transform to feature vectors
dataset = pd.read_csv('dataset.csv', sep='\t', header=None)
input_sentence_list = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))
vector = CountVectorizer()
vector.fit(input_sentence_list.values)
input_features_list = vector.transform(input_sentence_list.values)

# divide dataset
train_X, test_X, train_Y, test_Y = train_test_split(input_features_list, dataset[1].values, test_size=0.2, random_state=1037)

# load model and train model parameters
model1 = LogisticRegression()
model1.fit(train_X, train_Y)
model2 = SVC(kernel='linear')
model2.fit(train_X, train_Y)

# test model
predictions = model1.predict(test_X)
print('LogisticRegression results:', (np.array(test_Y) == np.array(predictions)).sum(), test_X.shape[0])
predictions = model2.predict(test_X)
print('SVM results:', (np.array(test_Y) == np.array(predictions)).sum(), test_X.shape[0])

#eval model
eval_quary = '请播放郭德纲小品'
eval_sentence = ' '.join(jieba.lcut(eval_quary))
eval_feature = vector.transform([eval_sentence])
print('LogisticRegression prediction:', model1.predict(eval_feature))
print('SVM prediction:', model2.predict(eval_feature))
