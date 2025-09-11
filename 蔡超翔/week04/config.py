REGEX_RULE = {
    "FilmTele-Play": ["播放", "电视剧"], # 句子是不是包含特定的单词，做出分类
    "HomeAppliance-Control": ["空调", "广播"]
}




CATEGORY_NAME = [
    'Travel-Query', 'Music-Play', 'FilmTele-Play', 'Video-Play',
    'Radio-Listen', 'HomeAppliance-Control', 'Weather-Query',
    'Alarm-Update', 'Calendar-Query', 'TVProgram-Play', 'Audio-Play',
    'Other'
]

TFIDF_MODEL_PKL_PATH = "../assets/weights/tfidf_ml.pkl"

# 训练后的模型路径（使用您提供的路径）
BERT_MODEL_PRETRAINED_PATH = "D:/ai/PycharmProjects/nlp20/week04/project1/training_code/food_review_bert/final_model"

LLM_OPENAI_SERVER_URL = f"http://127.0.0.1:11434/v1" # ollama
LLM_OPENAI_API_KEY = "None"
LLM_MODEL_NAME = "qwen2.5:0.5b"
