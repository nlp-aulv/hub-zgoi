from sklearn.neural_network import MLPClassifier
import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. 加载数据（假设dataset.csv是文本\t标签格式）之前老师讲到的是头部header不设置默认是第一行，现在设置为none后在设置一个名称
dataset = pd.read_csv("dataset.csv", sep="\t", header=None, names=["text", "label"])
print(dataset)
# 2. 定义分词函数
def chinese_tokenizer(text):
    return list(jieba.cut(text))  # 确保返回列表

# 3. 初始化TF-IDF向量化器，之前老师 用的是CountVectorizer仅统计了数量，在此处使用TfidfVectorizer，进行特征词提取，不转换小写，默认提取5万个特征词
vector = TfidfVectorizer(
    tokenizer=chinese_tokenizer,
    lowercase=False,
    max_features=50000  # 限制特征数量
)
vector.fit(dataset["text"])
# 4. 特征提取
X = vector.fit_transform(dataset["text"])

y = dataset["label"]

# 5. 训练MLP模型， 在此进行了训练
model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 两层隐藏层
    max_iter=500,
    random_state=42
)
model.fit(X, y)

# 6. 预测示例
test_text = "你真是个天才"
test_vec = vector.transform([test_text])
pred = model.predict(test_vec)
print(123)
print(f"'{test_text}' → {pred}")


# 机器学习sklearn的基本流程：
#1、加载数据模型 比如使用pandas的pd.read_csv()读取文件和进行处理
# 2、对数据清洗=>把数据转换成向量，使用TfidfVectorizer或者CountVectorizer，进行一次向量特征的（vector.fit)特征学习，fit在此处的作用是：建立数据集和避免泄露，初始化模型参数
# 应该先进行向量的训练再进行转换特征提取vector.fit_transform(dataset["text"])
# 3、训练模型，在已有框架下比如sklearn下很简单就是使用现成的模型：model = MLPClassifier(各种人工参数)。
# 4、选择好模型后使用model.fit模型学习，这里的模型训练需要输入和输出的一个数据集建立。
# 5、给定一个目标，test_text = "你真是个天才"。对目标值进行转换转换成模型识别的矩阵或者张量：test_vec = vector.transform([test_text])
# 对训练好的模型进行和目标结果进行结果输出：看是否能得到目标结果
# 6、输出模型的预测结果：model.predict(test_vec)


