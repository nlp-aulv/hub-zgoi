import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model # 线性模型模块
from sklearn import tree

dataset = pd.read_csv('dataset.csv',sep='\t',header=None)
# apply()是pandas对象的核心方法,用于对数据执行逐元素的操作,Series.apply(func) -> 对Series的每个元素应用函数func
processed_sentences = dataset[0].apply(lambda x: ' '.join(jieba.lcut(x)))

# 创建模型转换器 -> 学习词汇表 -> 文本转矩阵
vector = CountVectorizer()
vector.fit(processed_sentences.values)
input_feature = vector.transform(processed_sentences.values)

# 创建线性回归模型
line_model = linear_model.LogisticRegression()
line_model.fit(input_feature,dataset[1].values)
print(line_model)

test_line_txt = '我想去深圳'
test_line_sentence = ' '.join(jieba.lcut(test_line_txt))
test_line_feature = vector.transform([test_line_sentence])
print('线性回归预测结果:',line_model.predict(test_line_feature))

# 决策树模型
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(input_feature, dataset[1].values)
print(tree_model)

test_tree_txt = '明天是什么天气'
test_tree_sentence= ' '.join(jieba.lcut(test_tree_txt))
test_tree_feature = vector.transform([test_tree_sentence])
print('决策树模型预测结果',tree_model.predict(test_tree_feature))
