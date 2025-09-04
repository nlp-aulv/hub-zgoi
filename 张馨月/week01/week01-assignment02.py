DATA_PATH = os.path.join(os.path.dirname(__file__), "dataset.csv")

def load_dataset(path: str) -> pd.DataFrame:
    # 兼容常见的分隔符与编码
    for sep in ["\t", ",", ";"]:
        for enc in ["utf-8", "utf-8-sig", "gbk"]:
            try:
                df = pd.read_csv(path, sep=sep, header=None, encoding=enc)
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ["text", "label"]
                    return df
            except Exception:
                continue
    raise RuntimeError(
        "无法读取 dataset.csv，请确认文件位于脚本同目录；"
        "两列 [text, label]；分隔符为 Tab/逗号/分号；编码为 UTF-8 或 GBK。"
    )


def cut_text(s: str) -> str:
    return " ".join(jieba.lcut(str(s)))


def split_data(df: pd.DataFrame):
    # 分层抽样保护少数类（每类至少2个样本才可分层）
    label_counts = df["label"].value_counts()
    stratify = df["label"] if (df["label"].nunique() > 1 and label_counts.min() >= 2) else None
    return train_test_split(
        df["cut"],
        df["label"],
        test_size=0.2,
        random_state=520,
        stratify=stratify,
    )


def run_one(model_name, vectorizer, estimator, X_train_raw, X_test_raw, y_train, y_test):
    Xtr = vectorizer.fit_transform(X_train_raw)
    Xte = vectorizer.transform(X_test_raw)

    estimator.fit(Xtr, y_train)
    pred = estimator.predict(Xte)

    acc = accuracy_score(y_test, pred)
    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred, digits=4, zero_division=0))
    return acc


def main():
    print(f"读取数据：{DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"未找到 dataset.csv：{DATA_PATH}")

    df = load_dataset(DATA_PATH)
    print(f"样本数: {len(df)}，类别数: {df['label'].nunique()}")
    print("每类样本数（升序）：\n", df["label"].value_counts().sort_values())

    # 中文分词
    df["cut"] = df["text"].astype(str).apply(cut_text)

    # 切分
    X_train, X_test, y_train, y_test = split_data(df)

    # 模型 A：KNN + CountVectorizer（袋-of-words）
    run_one(
        "KNN + CountVectorizer",
        CountVectorizer(min_df=1),
        KNeighborsClassifier(n_neighbors=3),
        X_train, X_test, y_train, y_test,
    )

    # 模型 B：LogisticRegression（多分类）+ TF-IDF（带 ngram/阈值）
    run_one(
        "LogReg + TF-IDF (balanced, multinomial, ngram(1,2))",
        TfidfVectorizer(min_df=2, max_df=0.95, ngram_range=(1, 2)),
        LogisticRegression(
            solver="lbfgs",
            multi_class="multinomial",
            class_weight="balanced",
            max_iter=2000,
        ),
        X_train, X_test, y_train, y_test,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
