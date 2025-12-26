import pandas as pd
import numpy as np
from typing import List, Dict
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 定义自定义的提示模板
PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""


class GPT4RecMovieRecommender:
    def __init__(self, top_n_recommendations: int = 10):
        """
        初始化电影推荐器

        Args:
            top_n_recommendations: 最终推荐数量
        """
        self.top_n = top_n_recommendations
        self.movies_df = None
        self.ratings_df = None
        self.user_history = {}
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.movie_titles_set = set()  # 用于快速查找电影标题

    def load_data(self, ratings_path: str, movies_path: str):
        """
        加载电影评分和电影信息数据
        """
        # 加载评分数据
        self.ratings_df = pd.read_csv(
            ratings_path,
            sep="::",
            header=None,
            engine='python',
            encoding='latin'
        )
        self.ratings_df.columns = ["user_id", "movie_id", "rating", "timestamp"]

        # 加载电影信息
        self.movies_df = pd.read_csv(
            movies_path,
            sep="::",
            header=None,
            engine='python',
            encoding='latin'
        )
        self.movies_df.columns = ["movie_id", "title", "genres"]

        # 创建电影标题集合，用于快速查找
        self.movie_titles_set = set(self.movies_df['title'].tolist())

        # 构建用户历史观看记录
        self._build_user_history()

        # 构建TF-IDF向量用于检索
        self._build_tfidf_index()

    def _build_user_history(self):
        """构建用户历史观看记录"""
        # 筛选评分较高的电影（ >=7.5为喜欢）
        positive_ratings = self.ratings_df[self.ratings_df['rating'] >=7.5]

        # 按用户分组，获取每个用户看过的电影
        for user_id, group in positive_ratings.groupby('user_id'):
            movie_ids = group['movie_id'].tolist()
            # 获取电影标题
            titles = self.movies_df[self.movies_df['movie_id'].isin(movie_ids)]['title'].tolist()
            self.user_history[user_id] = titles[:10]  # 最多取10部

        print(f"Built history for {len(self.user_history)} users")

    def _build_tfidf_index(self):
        """构建TF-IDF检索索引"""
        # 使用电影标题+类型作为文本特征
        self.movies_df['content'] = self.movies_df['title'] + " " + self.movies_df['genres']

        print("Building TF-IDF index...")
        print(f"Sample content: {self.movies_df['content'].iloc[:3].tolist()}")

        # 创建TF-IDF向量器
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000
        )

        # 为所有电影生成TF-IDF向量
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.movies_df['content'])
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def generate_search_query(self, user_history_titles: List[str]) -> str:
        """
        模拟GPT-2生成搜索查询（简化版：基于规则生成）

        Args:
            user_history_titles: 用户历史观看的电影标题列表

        Returns:
            生成的搜索查询字符串
        """
        if not user_history_titles:
            return "popular movies"

        # 提取常见的关键词（这里简化处理，实际应用中可以使用NLP技术）
        all_titles = " ".join(user_history_titles).lower()

        # 简单的关键词提取（实际应用中可以使用TF-IDF或主题建模）
        keywords = {
            'action': ['action', 'adventure', 'fight', 'war', 'battle'],
            'comedy': ['comedy', 'funny', 'humor', 'laugh'],
            'drama': ['drama', 'romance', 'love', 'relationship'],
            'sci-fi': ['sci-fi', 'space', 'future', 'alien', 'robot'],
            'horror': ['horror', 'scary', 'ghost', 'monster', 'zombie'],
            'animation': ['animation', 'animated', 'cartoon', 'disney'],
            'crime': ['crime', 'police', 'detective', 'murder'],
            'fantasy': ['fantasy', 'magic', 'dragon', 'wizard']
        }

        # 检测主要类型
        detected_genres = []
        for genre, words in keywords.items():
            if any(word in all_titles for word in words):
                detected_genres.append(genre)

        # 如果没有检测到具体类型，返回通用查询
        if not detected_genres:
            # 从标题中提取前2个电影名称
            if len(user_history_titles) >= 2:
                query = f"movies similar to {user_history_titles[0]} and {user_history_titles[1]}"
            else:
                query = f"movies like {user_history_titles[0]}"
        else:
            # 基于检测到的类型生成查询
            if len(detected_genres) == 1:
                query = f"{detected_genres[0]} movies"
            else:
                # 随机选择1-2个类型来生成查询
                selected_genres = random.sample(detected_genres, min(2, len(detected_genres)))
                if len(selected_genres) == 1:
                    query = f"{selected_genres[0]} movies"
                else:
                    query = f"movies with {selected_genres[0]} and {selected_genres[1]} elements"

        return query

    def search_movies_by_query(self, query: str, top_k: int = 20) -> List[str]:
        """
        使用TF-IDF余弦相似度搜索电影（模拟BM25检索）

        Args:
            query: 搜索查询
            top_k: 返回的电影数量

        Returns:
            电影标题列表
        """
        try:
            # 将查询转换为TF-IDF向量
            query_vector = self.tfidf_vectorizer.transform([query])

            # 计算余弦相似度
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

            # 获取最相似的电影索引
            top_indices = similarities.argsort()[-top_k * 2:][::-1]  # 获取更多结果用于筛选

            # 返回电影标题
            recommended_titles = []
            for idx in top_indices:
                if similarities[idx] > 0.05:  # 降低相似度阈值
                    title = self.movies_df.iloc[idx]['title']
                    # 确保是有效的电影标题
                    if title in self.movie_titles_set:
                        recommended_titles.append(title)
                if len(recommended_titles) >= top_k:
                    break

            return recommended_titles[:top_k]
        except Exception as e:
            print(f"Error in search_movies_by_query: {e}")
            # 返回一个后备推荐列表
            return self.get_random_movies(top_k)

    def get_random_movies(self, n: int = 10) -> List[str]:
        """获取随机电影作为后备推荐"""
        random_movies = self.movies_df.sample(min(n, len(self.movies_df)))['title'].tolist()
        return random_movies

    def recommend_for_user(self, user_id: int) -> Dict:
        """
        为用户生成推荐

        Args:
            user_id: 用户ID

        Returns:
            包含推荐结果的字典
        """
        if user_id not in self.user_history:
            print(f"User {user_id} not found in history")
            # 返回默认推荐
            return {
                "user_id": user_id,
                "history_movies": [],
                "generated_queries": ["popular movies"],
                "recommendations": self.get_random_movies(self.top_n),
                "explanation": "Recommended popular movies"
            }

        # 获取用户历史
        history_titles = self.user_history[user_id]

        # 步骤1: 生成搜索查询（模拟GPT-2）
        search_query = self.generate_search_query(history_titles)
        print(f"Generated search query for user {user_id}: {search_query}")

        # 步骤2: 检索电影
        # 生成多个查询以获得多样性（模拟Beam Search）
        queries = [search_query]

        # 添加一些变体查询以增加多样性
        if len(history_titles) >= 2:
            # 基于具体电影名称的查询
            specific_query = f"movies like {history_titles[0]} and {history_titles[1]}"
            queries.append(specific_query)

        # 还可以添加基于类型的查询变体
        if 'movies with' in search_query:
            # 尝试提取类型信息
            if 'action' in search_query.lower():
                queries.append("action adventure movies")
            if 'comedy' in search_query.lower():
                queries.append("comedy films")

        print(f"Generated queries: {queries}")

        # 检索结果
        all_recommendations = []
        for query in queries:
            try:
                recommendations = self.search_movies_by_query(query, top_k=self.top_n // len(queries) + 2)
                print(f"Query '{query}' found {len(recommendations)} recommendations")
                all_recommendations.extend(recommendations)
            except Exception as e:
                print(f"Error searching for query '{query}': {e}")

        # 去重并限制数量
        unique_recommendations = []
        seen = set()
        for movie in all_recommendations:
            if movie not in seen and movie not in history_titles:
                seen.add(movie)
                unique_recommendations.append(movie)
            if len(unique_recommendations) >= self.top_n:
                break

        # 如果推荐结果不足，补充随机电影
        if len(unique_recommendations) < self.top_n:
            needed = self.top_n - len(unique_recommendations)
            random_movies = self.get_random_movies(needed * 2)  # 获取多一些以防重复
            for movie in random_movies:
                if movie not in seen and movie not in history_titles:
                    seen.add(movie)
                    unique_recommendations.append(movie)
                if len(unique_recommendations) >= self.top_n:
                    break

        # 准备结果
        result = {
            "user_id": user_id,
            "history_movies": history_titles,
            "generated_queries": queries,
            "recommendations": unique_recommendations[:self.top_n],
            "explanation": f"Recommended based on your interest in: {search_query}"
        }

        return result

    def format_recommendation_prompt(self, user_id: int) -> str:
        """
        生成推荐提示（使用提供的模板）
        """
        if user_id not in self.user_history:
            return "User not found"

        history_titles = self.user_history[user_id]
        history_str = "\n".join([f"- {title}" for title in history_titles])

        prompt = PROMPT_TEMPLATE.format(history_str)
        return prompt


# 使用示例
if __name__ == "__main__":
    # 初始化推荐器
    recommender = GPT4RecMovieRecommender(top_n_recommendations=10)

    # 加载数据
    ratings_path = "./M_ML-100K/ratings.dat"
    movies_path = "./M_ML-100K/movies.dat"

    try:
        recommender.load_data(ratings_path, movies_path)
        print("Data loaded successfully!")
        print(f"Total users: {len(recommender.user_history)}")
        print(f"Total movies: {len(recommender.movies_df)}")

        # 测试为用户推荐 - 尝试多个用户
        for test_user_id in [1, 2, 3, 10, 50]:
            if test_user_id in recommender.user_history:
                print("\n" + "=" * 50)
                print(f"Recommendations for User {test_user_id}:")
                print("=" * 50)

                # 获取推荐
                recommendations = recommender.recommend_for_user(test_user_id)

                print(f"History ({len(recommendations['history_movies'])} movies):")
                for movie in recommendations['history_movies'][:3]:  # 只显示前3部
                    print(f"  - {movie}")
                if len(recommendations['history_movies']) > 3:
                    print(f"  ... and {len(recommendations['history_movies']) - 3} more")

                print(f"\nGenerated Queries: {recommendations['generated_queries']}")

                print(f"\nTop {recommender.top_n} Recommendations:")
                for i, movie in enumerate(recommendations['recommendations'], 1):
                    print(f"  {i:2d}. {movie}")

                # 生成提示文本
                print("\n" + "=" * 50)
                print("Generated Prompt for LLM:")
                print("=" * 50)
                prompt = recommender.format_recommendation_prompt(test_user_id)
                print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
                print("\n")
                break  # 只测试第一个有效用户

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")
        print("Please make sure the data files are in the correct path.")
        print("Current paths:")
        print(f"  Ratings: {ratings_path}")
        print(f"  Movies: {movies_path}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()