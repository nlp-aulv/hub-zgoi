from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from DataLoadUtils import DataLoadUtils

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = DataLoadUtils(r"./dataset.csv").dataset()

    max_test_acc = -1
    max_acc_n_neighbor = 1
    # KNN预测
    for i in range(1, 11):
        print("n_neighbors = {}".format(i))
        model = KNeighborsClassifier(n_neighbors=i)
        model.fit(X_train, y_train)
        y_predict_trian = model.predict(X_train)
        print("训练集准确率：{:.2f} %".format(accuracy_score(y_predict_trian, y_train.values) * 100))

        y_predict_test = model.predict(X_test)
        test_acc = accuracy_score(y_predict_test, y_test.values) * 100
        print("测试集准确率：{:.2f} %".format(test_acc))
        print("=" * 20)

        if test_acc > max_test_acc:
            max_test_acc = test_acc
            max_acc_n_neighbor = i

    print("测试集最大准确率出现在 k = {}， 准确率：{:.2f} %".format(max_acc_n_neighbor, max_test_acc))
