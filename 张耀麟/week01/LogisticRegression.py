from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from DataLoadUtils import DataLoadUtils

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = DataLoadUtils(r"./dataset.csv").dataset()

    # 逻辑回归
    print("solver = lbfgs (default)")
    model = LogisticRegression(max_iter=50)
    model.fit(X_train, y_train)
    y_predict_trian = model.predict(X_train)
    print("训练集准确率：{:.2f} %".format(accuracy_score(y_predict_trian, y_train.values) * 100))

    y_predict_test = model.predict(X_test)
    print("测试集准确率：{:.2f} %".format(accuracy_score(y_predict_test, y_test.values) * 100))
    print("=" * 20)

    # 逻辑回归
    print("solver = liblinear")
    model2 = LogisticRegression(max_iter=25, solver="liblinear")
    model2.fit(X_train, y_train)
    y_predict_trian = model2.predict(X_train)
    print("训练集准确率：{:.2f} %".format(accuracy_score(y_predict_trian, y_train.values) * 100))

    y_predict_test = model2.predict(X_test)
    print("测试集准确率：{:.2f} %".format(accuracy_score(y_predict_test, y_test.values) * 100))
