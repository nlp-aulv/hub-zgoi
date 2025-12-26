import pandas as pd
import openai
import os
from openai import OpenAI

os.environ["OPENAI_API_KEY"] = "sk-XXXXXXXX"

# 读取数据
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding='latin-1')
movies.columns = ['movie_id', 'movie_title', 'movie_tag']

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请根据用户观看电影的偏好，给用户推荐可能喜欢的电影。
如下是历史观看的电影:
{0}
请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐，格式如下：
电影名称 - 类型 - 简短描述(描述不超过100字)
"""

def get_user_watched_movies(user_id, top_n=10):
    """获取用户观看过的电影"""
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False).head(top_n)
    watched_movie_ids = user_ratings['movie_id'].tolist()
    watched_movies = movies[movies['movie_id'].isin(watched_movie_ids)]
    return watched_movies, watched_movie_ids

def call_llm_for_recommendations(watched_movies_text):
    client = OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    prompt = PROMPT_TEMPLATE.format(watched_movies_text)

    response = client.chat.completions.create(
        model="qwen-max",
        messages=[
            {"role": "system", "content": "你是一个电影推荐专家。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )

    return response.choices[0].message.content

def search_movies_in_database(llm_recommendations, watched_movie_ids):
    recommendations_lines = llm_recommendations.strip().split('\n')

    recommended_movies = []

    for line in recommendations_lines:
        if not line.strip():
            continue

        # 提取电影关键词（取第一个 - 之前的内容作为电影名）
        movie_keywords = line.split('-')[0].strip()

        # 在数据库中搜索相似的电影（排除已观看的）
        available_movies = movies[movies['movie_id'].isin(watched_movie_ids)]

        # 模糊匹配电影标题
        matched = available_movies[
            available_movies['movie_title'].str.contains(movie_keywords, case=False, na=False, regex=False)
        ]

        if len(matched) > 0:
            recommended_movies.append(matched.iloc[0])
        else:
            # 如果没有精确匹配，尝试匹配关键词中的单词
            keywords = movie_keywords.split()
            for keyword in keywords:
                if len(keyword) > 3:  # 只使用长度大于3的关键词
                    matched = available_movies[available_movies['movie_title'].str.contains(keyword, case=False, na=False, regex=False)]
                    if len(matched) > 0:
                        recommended_movies.append(matched.iloc[0])
                        break

    return pd.DataFrame(recommended_movies).drop_duplicates()

def recommend_movies_for_user(user_id):
    """为用户推荐电影的流程"""

    # 1. 获取用户观看过的电影
    watched_movies, watched_movie_ids = get_user_watched_movies(user_id, top_n=10)
    print("用户历史观看的电影（10部）：")
    for idx, row in watched_movies.iterrows():
        print(f"  - {row['movie_title']} ({row['movie_tag']})")

    # 2. 构建观看历史文本
    watched_movies_text = "\n".join([
        f"{row['movie_title']} - {row['movie_tag']}"
        for idx, row in watched_movies.iterrows()
    ])

    # 3. 调用大语言模型获取推荐
    print("\n正在调用大语言模型生成推荐...")
    llm_recommendations = call_llm_for_recommendations(watched_movies_text)
    print("\n大语言模型推荐结果：")
    print(llm_recommendations)

    # 4. 在数据库中检索推荐的电影
    print("\n在电影数据库中检索相关电影...")
    recommended_movies = search_movies_in_database(llm_recommendations, watched_movie_ids)

    # 5. 输出最终推荐结果
    print("最终推荐结果：")
    if len(recommended_movies) > 0:
        for idx, row in recommended_movies.iterrows():
            print(f"  {row['movie_id']}. {row['movie_title']} - {row['movie_tag']}")
    else:
        print("  未找到匹配的电影，请尝试其他用户或调整推荐策略。")

    return recommended_movies

# 主程序
if __name__ == "__main__":
    recommended_movies = recommend_movies_for_user(196)
