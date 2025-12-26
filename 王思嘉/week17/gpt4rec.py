import pandas as pd
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# 读取数据
ratings = pd.read_csv("Week17/03_推荐系统/M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]

movies = pd.read_csv("Week17/03_推荐系统/M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]  # 修正列名

PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
"""

class GPT4Rec:
    def __init__(self, api_key=None):
        """
        初始化GPT4Rec推荐系统
        :param api_key: OpenAI API密钥
        """
        if api_key:
            openai.api_key = api_key
        self.movies_df = movies
        self.ratings_df = ratings
        
    def get_user_history(self, user_id, n_items=10):
        """
        获取用户历史交互的物品
        :param user_id: 用户ID
        :param n_items: 返回历史物品数量
        :return: 历史物品列表
        """
        # 获取用户评分过的电影
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        # 按时间戳排序，获取最近的n_items个
        user_ratings = user_ratings.sort_values('timestamp', ascending=False).head(n_items)
        
        # 合并电影标题信息
        user_movies = pd.merge(user_ratings, self.movies_df, on='movie_id')
        return user_movies['movie_title'].tolist()
    
    def generate_query(self, history_items):
        """
        使用GPT生成用户未来可能的查询
        :param history_items: 用户历史交互物品列表
        :return: 生成的查询列表
        """
        history_text = "\n".join(history_items)
        prompt = PROMPT_TEMPLATE.format(history_text)
        
        try:
            # 调用OpenAI API生成查询
            response = openai.Completion.create(
                engine="text-davinci-003",  # 或使用gpt-3.5-turbo
                prompt=prompt,
                max_tokens=200,
                n=1,
                stop=None,
                temperature=0.7
            )
            
            generated_text = response.choices[0].text.strip()
            # 解析生成的推荐电影
            recommendations = [line.strip() for line in generated_text.split('\n') if line.strip()]
            return recommendations
        except Exception as e:
            print(f"API调用失败: {e}")
            return []
    
    def retrieve_items(self, queries, top_k=10):
        """
        根据生成的查询检索相关物品
        :param queries: 生成的查询列表
        :param top_k: 检索的物品数量
        :return: 检索到的物品列表
        """
        if not queries:
            # 如果没有生成查询，使用TF-IDF基于历史记录推荐
            all_titles = self.movies_df['movie_title'].tolist()
            return all_titles[:top_k]
        
        # 使用TF-IDF向量化电影标题和查询
        vectorizer = TfidfVectorizer()
        
        # 合并查询和电影标题
        all_texts = queries + self.movies_df['movie_title'].tolist()
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # 计算查询与所有电影的相似度
        query_vectors = tfidf_matrix[:len(queries)]
        movie_vectors = tfidf_matrix[len(queries):]
        
        # 计算相似度
        similarities = cosine_similarity(query_vectors, movie_vectors)
        
        # 合并所有查询的相似度得分
        avg_similarities = similarities.mean(axis=0)
        
        # 获取最相似的电影索引
        top_indices = avg_similarities.argsort()[-top_k:][::-1]
        
        # 返回推荐的电影
        recommended_movies = self.movies_df.iloc[top_indices]['movie_title'].tolist()
        return recommended_movies
    
    def recommend(self, user_id, n_history=5, top_k=10):
        """
        为用户生成推荐
        :param user_id: 用户ID
        :param n_history: 使用的历史交互数量
        :param top_k: 推荐物品数量
        :return: 推荐物品列表
        """
        # 步骤1：获取用户历史
        history_items = self.get_user_history(user_id, n_history)
        print(f"用户历史电影: {history_items}")
        
        # 步骤2：生成查询条件
        generated_queries = self.generate_query(history_items)
        print(f"生成的查询: {generated_queries}")
        
        # 步骤3：检索相关物品
        recommendations = self.retrieve_items(generated_queries, top_k)
        
        return recommendations

# 使用示例
def main():
    # 初始化推荐系统（需要设置OpenAI API密钥）
    gpt4rec = GPT4Rec(api_key="your-openai-api-key-here")
    
    # 为用户1生成推荐
    user_id = 1
    recommendations = gpt4rec.recommend(user_id=user_id, n_history=5, top_k=10)
    
    print(f"为用户 {user_id} 推荐的电影:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie}")

if __name__ == "__main__":
    main()
