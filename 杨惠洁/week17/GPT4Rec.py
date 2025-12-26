import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import random

# 读取数据
ratings = pd.read_csv("./M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("./M_ML-100K/movies.dat", sep="::", header=None, engine='python')
movies.columns = ["movie_id", "movie_title", "movie_tag"]

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可以观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选电影描述，每一行是一个推荐
"""

# 查看数据结构
print("评分数据 shape:", ratings.shape)
print("电影数据 shape:", movies.shape)
print("\n电影数据前5行:")
print(movies.head())
print("\n评分数据前5行:")
print(ratings.head())

# 合并数据，创建用户-电影交互矩阵
movie_info = movies.set_index('movie_id')[['movie_title', 'movie_tag']]
ratings_with_info = ratings.join(movie_info, on='movie_id', how='left')


# 数据预处理函数
def preprocess_text(text):
    """预处理文本"""
    if pd.isna(text):
        return ""
    # 移除年份和特殊字符
    text = re.sub(r'\(\d{4}\)', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = text.lower().strip()
    return text


# 预处理电影标题和标签
movies['processed_title'] = movies['movie_title'].apply(preprocess_text)
movies['processed_tag'] = movies['movie_tag'].apply(lambda x: x.replace('|', ' ') if pd.notna(x) else "")

# 构建电影内容特征
movies['content'] = movies['processed_title'] + ' ' + movies['processed_tag']

# 创建TF-IDF向量化器
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
movie_vectors = tfidf.fit_transform(movies['content'])
movie_id_to_idx = {movie_id: idx for idx, movie_id in enumerate(movies['movie_id'])}


# 基于规则的查询生成函数（模拟GPT的查询生成）
def generate_queries_from_history(movie_titles, n_queries=5):
    """
    从用户历史电影生成搜索查询
    这是基于规则的简化版本，实际GPT4Rec使用GPT-2生成
    """
    queries = []

    # 提取电影中的关键词
    all_words = []
    for title in movie_titles:
        words = preprocess_text(title).split()
        all_words.extend(words)

    # 统计词频
    word_counts = Counter(all_words)

    # 生成查询策略1：基于高频词
    if word_counts:
        top_words = [word for word, count in word_counts.most_common(3) if len(word) > 3]
        if top_words:
            queries.append(' '.join(top_words))

    # 生成查询策略2：组合电影类型
    tags = set()
    for title in movie_titles:
        movie_row = movies[movies['movie_title'] == title]
        if not movie_row.empty:
            tag_str = movie_row.iloc[0]['movie_tag']
            if pd.notna(tag_str):
                tags.update(tag_str.split('|'))

    if tags:
        queries.append(' '.join(list(tags)[:3]))

    # 生成查询策略3：基于电影标题的模式
    for title in movie_titles[:min(3, len(movie_titles))]:
        words = preprocess_text(title).split()
        if len(words) >= 2:
            queries.append(' '.join(words[:2]))

    # 如果查询数量不够，添加通用查询
    while len(queries) < n_queries:
        if word_counts:
            random_words = random.sample([w for w in word_counts.keys() if len(w) > 3],
                                         min(3, len(word_counts)))
            queries.append(' '.join(random_words))
        else:
            queries.append('movie film')

    return list(set(queries))[:n_queries]


# 检索函数
def retrieve_movies(query, top_k=10, movie_vectors=movie_vectors, movies_df=movies):
    """基于查询检索电影"""
    # 将查询转换为向量
    query_vec = tfidf.transform([query])

    # 计算相似度
    similarities = cosine_similarity(query_vec, movie_vectors).flatten()

    # 获取top_k电影
    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        if similarities[idx] > 0:
            movie_id = movies_df.iloc[idx]['movie_id']
            movie_title = movies_df.iloc[idx]['movie_title']
            similarity = similarities[idx]
            results.append({
                'movie_id': movie_id,
                'movie_title': movie_title,
                'similarity': similarity
            })

    return results


# GPT4Rec风格推荐函数
def gpt4rec_recommend(user_id, ratings_df=ratings, movies_df=movies, top_k=10, n_queries=5):
    """
    模拟GPT4Rec的推荐流程：
    1. 获取用户历史电影
    2. 生成搜索查询
    3. 基于查询检索电影
    4. 合并结果
    """
    # 1. 获取用户历史电影（高评分电影）
    user_ratings = ratings_df[ratings_df['user_id'] == user_id]
    user_ratings = user_ratings.sort_values('rating', ascending=False)

    # 获取用户高评分电影（rating >= 4）
    high_rated = user_ratings[user_ratings['rating'] >= 4]
    if len(high_rated) == 0:
        high_rated = user_ratings.head(5)  # 如果没有高评分，取最近看的5部

    # 获取电影标题
    movie_titles = []
    for _, row in high_rated.iterrows():
        movie_row = movies_df[movies_df['movie_id'] == row['movie_id']]
        if not movie_row.empty:
            movie_titles.append(movie_row.iloc[0]['movie_title'])

    if not movie_titles:
        print(f"用户 {user_id} 没有观看历史")
        return []

    print(f"用户 {user_id} 的历史电影：")
    for i, title in enumerate(movie_titles[:10], 1):
        print(f"{i}. {title}")

    # 2. 生成查询（模拟GPT-2生成）
    print(f"\n生成的查询（{n_queries}个）：")
    queries = generate_queries_from_history(movie_titles, n_queries=n_queries)
    for i, query in enumerate(queries, 1):
        print(f"{i}. '{query}'")

    # 3. 基于每个查询检索电影
    all_results = {}
    for query in queries:
        results = retrieve_movies(query, top_k=top_k * 2)  # 每个查询多检索一些
        for result in results:
            movie_id = result['movie_id']
            if movie_id not in all_results:
                all_results[movie_id] = {
                    'movie_title': result['movie_title'],
                    'similarities': [result['similarity']],
                    'query_count': 1
                }
            else:
                all_results[movie_id]['similarities'].append(result['similarity'])
                all_results[movie_id]['query_count'] += 1

    # 4. 计算综合分数并排序
    scored_results = []
    for movie_id, info in all_results.items():
        # 过滤用户已经看过的电影
        if movie_id in user_ratings['movie_id'].values:
            continue

        # 综合评分：平均相似度 * 查询覆盖度
        avg_sim = np.mean(info['similarities'])
        query_coverage = info['query_count'] / n_queries
        final_score = avg_sim * (1 + 0.5 * query_coverage)  # 查询覆盖度越高，分数越高

        scored_results.append({
            'movie_id': movie_id,
            'movie_title': info['movie_title'],
            'score': final_score,
            'query_count': info['query_count']
        })

    # 5. 按分数排序并返回top_k
    scored_results.sort(key=lambda x: x['score'], reverse=True)

    return scored_results[:top_k]


# 使用Prompt模板进行推荐
def recommend_with_prompt(user_id, top_k=10):
    """使用Prompt模板格式进行推荐"""
    # 获取用户历史
    user_ratings = ratings[ratings['user_id'] == user_id]
    if len(user_ratings) == 0:
        return "用户不存在或没有观看历史"

    user_ratings = user_ratings.sort_values('timestamp', ascending=False)
    history_titles = []
    for _, row in user_ratings.head(10).iterrows():
        movie_row = movies[movies['movie_id'] == row['movie_id']]
        if not movie_row.empty:
            history_titles.append(movie_row.iloc[0]['movie_title'])

    # 生成Prompt
    history_str = "\n".join([f"- {title}" for title in history_titles])
    prompt = PROMPT_TEMPLATE.format(history_str)

    print("=" * 50)
    print("生成的Prompt:")
    print(prompt)
    print("=" * 50)

    # 使用GPT4Rec风格推荐
    recommendations = gpt4rec_recommend(user_id, top_k=top_k)

    print(f"\n为用户 {user_id} 推荐的电影：")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['movie_title']} (分数: {rec['score']:.4f}, 被{rec['query_count']}个查询命中)")

    return recommendations


# 评估函数
def evaluate_recommendations(user_id, test_size=5):
    """
    简单的评估函数：将用户最后几部电影作为测试集
    """
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('timestamp')
    if len(user_ratings) < test_size + 5:
        print(f"用户 {user_id} 的历史数据不足")
        return None

    # 分割训练集和测试集
    train_ratings = user_ratings.iloc[:-test_size]
    test_ratings = user_ratings.iloc[-test_size:]

    # 临时修改全局ratings用于训练
    global ratings
    original_ratings = ratings.copy()

    # 使用训练集
    train_global_ratings = ratings[~ratings.index.isin(test_ratings.index)]

    # 获取推荐
    recommendations = gpt4rec_recommend(user_id, ratings_df=train_global_ratings, top_k=20)
    recommended_ids = [rec['movie_id'] for rec in recommendations]

    # 计算Recall
    test_movie_ids = test_ratings['movie_id'].tolist()
    hits = len(set(recommended_ids) & set(test_movie_ids))
    recall = hits / len(test_movie_ids)

    # 恢复原始数据
    ratings = original_ratings

    print(f"\n评估结果 - 用户 {user_id}:")
    print(f"测试电影: {test_movie_ids}")
    print(f"命中数量: {hits}/{len(test_movie_ids)}")
    print(f"Recall@{20}: {recall:.4f}")

    return recall


# 主程序
if __name__ == "__main__":
    # 测试推荐系统
    print("电影推荐系统 - GPT4Rec风格实现")
    print("=" * 50)

    # 选择一个用户进行测试
    user_ids = ratings['user_id'].unique()
    test_user = user_ids[0]  # 使用第一个用户
    print(f"测试用户ID: {test_user}")

    # 方法1: 使用Prompt模板
    print("\n方法1: 使用Prompt模板推荐")
    recommendations = recommend_with_prompt(test_user, top_k=10)

    # 方法2: 直接使用GPT4Rec风格
    print("\n" + "=" * 50)
    print("方法2: 直接GPT4Rec风格推荐")
    recommendations2 = gpt4rec_recommend(test_user, top_k=10)

    # 评估推荐效果
    print("\n" + "=" * 50)
    print("推荐效果评估")
    recall = evaluate_recommendations(test_user, test_size=5)

    # 显示电影库统计信息
    print("\n" + "=" * 50)
    print("数据集统计信息:")
    print(f"用户数量: {len(user_ids)}")
    print(f"电影数量: {len(movies)}")
    print(f"评分数量: {len(ratings)}")
    print(f"平均每个用户评分: {len(ratings) / len(user_ids):.2f}")

    # 查看不同类型的电影示例
    print("\n不同类别的电影示例:")
    genres = set()
    for tags in movies['movie_tag'].dropna():
        genres.update(tags.split('|'))

    print(f"电影类别总数: {len(genres)}")
    print("部分类别:", list(genres)[:10])