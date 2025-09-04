import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer


class DataLoadUtils:
    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path, header=None, sep='\t')
        # print("show data:\n", data.head(10))

    def dataset(self, test_size=0.3) -> tuple:
        data = self.data
        # 中文语句分词
        X = data[0].apply(lambda x: " ".join(jieba.lcut(x)))
        y = data[1]

        # 数据集分割，训练集：测试集 = 7：3
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

        # print(X_train.head(10))

        # 对文本进行提取特征 默认是使用标点符号分词
        count_vectorizer = CountVectorizer()
        count_vectorizer.fit(X_train.values)
        train_input_feature = count_vectorizer.transform(X_train.values)
        test_input_feature = count_vectorizer.transform(X_test.values)

        return train_input_feature, test_input_feature, y_train, y_test
