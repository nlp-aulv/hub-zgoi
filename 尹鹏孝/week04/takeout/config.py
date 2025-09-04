# 我按照0，1匹配输入一句话判断1是好评，0是差评。0,1来源于数据集的标识，目前没有对数据集做出清洗，有些数据集集对好评和差评的判断是错误的。
CATEGORY_NAME = [
    '1', '0'
]
BERT_MODEL_PKL_PATH = "assets/weights/bert.pkl"
BERT_MODEL_PERTRAINED_PATH = "assets/models/google-bert/bert-base-chinese"
BERT_DATA_PATH = "assets/dataset/作业数据-waimai_10k-change.csv"