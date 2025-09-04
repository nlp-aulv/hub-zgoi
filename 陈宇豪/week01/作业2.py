import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier

# 分割数据集
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


# 朴素贝叶斯
def nb_mode_predict(x_train, x_test, y_train, y_test):
    nb_model = MultinomialNB()
    nb_model.fit(x_train, y_train)
    # 预测
    y_pred = nb_model.predict(x_test)
    accuracy_nb = accuracy_score(y_test, y_pred)
    print("朴素贝叶斯正确率:{}%".format(np.round(accuracy_nb * 100, 2)))
    report_nb = classification_report(y_test, y_pred)
    print("朴素贝叶斯模型信息:\n{}".format(report_nb))
    return accuracy_nb, report_nb


# SVM支持向量机
from sklearn.svm import SVC


def svm_mode_predict(x_train, x_test, y_train, y_test):
    svm_model = SVC(decision_function_shape='ovo')
    svm_model.fit(x_train, y_train)
    y_pred_svm = svm_model.predict(x_test)
    # 预测
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM支持向量机正确率:{}%".format(np.round(accuracy_svm * 100, 2)))
    report_svm = classification_report(y_test, y_pred_svm)
    print("SVM支持向量机模型信息:\n{}".format(report_svm))
    return accuracy_svm, report_svm


# 随机森林
from sklearn.ensemble import RandomForestClassifier
import warnings

warnings.filterwarnings("ignore")


def forest_predict(x_train, x_test, y_train, y_test):
    forest_model = RandomForestClassifier(n_estimators=50)
    forest_model.fit(x_train, y_train)
    y_pred_forest = forest_model.predict(x_test)
    # 预测
    accuracy_forest = accuracy_score(y_test, y_pred_forest)
    print("随机森林正确率:{}%".format(np.round(accuracy_forest * 100, 2)))
    report_forest = classification_report(y_test, y_pred_forest)
    print("随机森林模型信息:\n{}".format(report_forest))
    return accuracy_forest, report_forest


# knn算法
from sklearn.neighbors import KNeighborsClassifier


def knn_predict(x_train, x_test, y_train, y_test):
    n_neighbors = [3, 5, 7]
    model_knn = None
    max_accuracy = 0
    for neighbor in n_neighbors:
        knn_model = KNeighborsClassifier(n_neighbors=neighbor)
        knn_model.fit(x_train, y_train)
        y_pred_knn = knn_model.predict(x_test)
        accuracy_knn = accuracy_score(y_test, y_pred_knn)
        print("邻居个数为{}的KNN正确率:{}%".format(neighbor, np.round(accuracy_knn * 100, 2)))
        if accuracy_knn > max_accuracy:
            max_accuracy = accuracy_knn
            model_knn = knn_model
    y_pred_knn = model_knn.predict(x_test)
    accuracy_knn_opt = accuracy_score(y_test, y_pred_knn)
    report_knn = classification_report(y_test, y_pred_knn)
    print("最优knn的模型信息:\n{}".format(report_knn))
    return accuracy_knn_opt, report_knn


if __name__ == "__main__":
    """
    去除无用词
    """
    # 给特征去除一些没用的词
    from stopwordsiso import stopwords

    stopwords = stopwords('zh')

    # 固定统一随机种子
    rand_seed = 22

    # 读取数据集
    dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
    print("数据集信息:\n{}".format(dataset.info()))
    print("数据集前5条样式:{}".format(dataset.head(5)))

    # jieba分词
    words_sentence = dataset[0].apply(lambda x: " ".join(jieba.lcut(x)))
    print("分词样式:{}".format(words_sentence[0]))

    # 对文本进行提取特征 默认是使用标点符号分词
    vector = CountVectorizer()
    vector.fit(words_sentence.values)
    # print(words_sentence.values)

    # 将分词转成特征向量
    input_feature = vector.transform(words_sentence.values)


    # 去除无用词列表
    def clean_text(words) -> str:
        if not words or pd.isna(words):
            return ''
        word_list = words.split()
        filtered_words = [word for word in word_list if word not in stopwords]
        return " ".join(filtered_words)


    words_sentence_stopword = words_sentence.apply(clean_text)

    # 对文本进行提取特征 默认是使用标点符号分词
    vector = CountVectorizer()
    vector.fit(words_sentence_stopword.values)

    # 将分词转成特征向量
    input_feature_stopword = vector.transform(words_sentence_stopword.values)
    x_train, x_test, y_train, y_test = train_test_split(input_feature, dataset[1], test_size=0.2,
                                                        random_state=rand_seed)
    x_train_stopword, x_test_stopword, y_train_stopword, y_test_stopword = train_test_split(input_feature_stopword,
                                                                                            dataset[1], test_size=0.2,
                                                                                            random_state=rand_seed)
    # 朴素贝叶斯
    accuracy_nb, report_nb = nb_mode_predict(x_train, x_test, y_train, y_test)
    accuracy_nb_stopword, report_nb_stopword = nb_mode_predict(x_train_stopword, x_test_stopword, y_train_stopword,
                                                               y_test_stopword)
    # SVM支持向量机
    accuracy_svm, report_svm = svm_mode_predict(x_train, x_test, y_train, y_test)
    accuracy_svm_stopword, report_svm_stopword = svm_mode_predict(x_train_stopword, x_test_stopword, y_train_stopword,
                                                                  y_test_stopword)
    # 随机森林
    accuracy_forest, report_forest = forest_predict(x_train, x_test, y_train, y_test)
    accuracy_forest_stopword, report_forest_stopword = forest_predict(x_train_stopword, x_test_stopword,
                                                                      y_train_stopword, y_test_stopword)
    # knn算法
    accuracy_knn, report_knn = knn_predict(x_train, x_test, y_train, y_test)
    accuracy_knn_stopword, report_knn_stopword = knn_predict(x_train_stopword, x_test_stopword, y_train_stopword,
                                                             y_test_stopword)

    """
    画图比较
    """
    import matplotlib.pyplot as plt

    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    plt.figure(figsize=(10, 6))
    x_label = ["贝叶斯", "svm", "随机森林", "KNN"]
    color_list = ['#FF6B6B', '#4ECDC4', '#45B7D1', "#45B7D1"]
    acc_list = [accuracy_nb, accuracy_svm, accuracy_forest, accuracy_knn]
    acc_list_stopword = [accuracy_nb_stopword, accuracy_svm_stopword, accuracy_forest_stopword, accuracy_knn]

    # 设置柱状图参数
    x = np.arange(len(x_label))  # 标签位置
    width = 0.35  # 柱子宽度

    # 创建图形
    plt.figure(figsize=(12, 8))

    # 创建分组柱状图
    bars1 = plt.bar(x - width / 2, acc_list, width,
                    label='正常', color='#FF6B6B', alpha=0.8, edgecolor='black')
    bars2 = plt.bar(x + width / 2, acc_list_stopword, width,
                    label='使用停用词', color='#4ECDC4', alpha=0.8, edgecolor='black')

    # 添加标题和标签
    plt.title('不同算法准确率对比（正常 vs 停用词处理）', fontsize=16, fontweight='bold')
    plt.xlabel('算法类型', fontsize=14)
    plt.ylabel('准确率', fontsize=14)
    plt.xticks(x, x_label)

    # 添加图例
    plt.legend(fontsize=12)


    # 为每个柱子添加数值标签
    def add_value_labels(bars, values):
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                     f'{value:.3f}', ha='center', va='bottom',
                     fontsize=11, fontweight='bold')


    # 添加数值标签
    add_value_labels(bars1, acc_list)
    add_value_labels(bars2, acc_list_stopword)

    # 设置y轴范围
    plt.ylim(0, max(max(acc_list), max(acc_list_stopword)) + 0.05)

    # 添加网格
    plt.grid(axis='y', alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.show()
