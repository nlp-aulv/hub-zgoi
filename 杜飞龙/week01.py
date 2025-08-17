def week01_func():
    import jieba
    import pandas as pd
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.metrics import accuracy_score

    # 1. 读取数据
    dataset = pd.read_csv("workWeek1/dataset.csv", sep="\t", header=None)
    print(dataset.head(5))

    # 2. 分词
    input_sentences = dataset[0].apply(lambda x: " ".join(jieba.lcut(str(x))))

    # 3. 特征提取
    vector = CountVectorizer()
    vector.fit(input_sentences.values)
    input_features = vector.transform(input_sentences.values)

    # 4. 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        input_features, dataset[1].values, test_size=0.25, random_state=42
    )

    # 5. KNN 分类
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    knn_pred = knn.predict(X_test)
    print("KNN准确率：", accuracy_score(y_test, knn_pred))

    # 6. 逻辑回归
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    print("逻辑回归准确率：", accuracy_score(y_test, lr_pred))

    # 7. 决策树
    dt = DecisionTreeClassifier()
    dt.fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    print("决策树准确率：", accuracy_score(y_test, dt_pred))

    # 8. 测试自定义文本
    custom_tests = [
        "帮我播放一下郭德纲的小品",
        "明天北京的天气怎么样",
        "给我定一个明天早上7点的闹钟",
        "我要听周杰伦的歌",
        "查询广州到上海的机票",
        "播放一部最新的美国电影",
        "把客厅的空调打开",
        "今年春节是几号",
        "我要看动漫全职高手",
        "收听中央人民广播电台"
    ]

    for test_query in custom_tests:
        test_sentence = " ".join(jieba.lcut(test_query))
        test_feature = vector.transform([test_sentence])
        print(f"\n测试文本: {test_query}")
        print("KNN模型预测结果: ", knn.predict(test_feature)[0])
        print("逻辑回归预测结果: ", lr.predict(test_feature)[0])
        print("决策树预测结果: ", dt.predict(test_feature)[0])

if __name__ == '__main__':
    week01_func()
