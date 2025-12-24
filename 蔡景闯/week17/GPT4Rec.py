import pandas as pd
from openai import OpenAI
from typing import List
import re
from rank_bm25 import BM25Okapi

def get_user_data(data_path:str) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep="::", header=None, names=["user_id", "movie_id", "rating", "timestamp"], engine="python")
    return data

def get_movies_data(data_path:str) -> pd.DataFrame:
    data = pd.read_csv(data_path, sep="::", header=None, names=["id", "title", "class"], engine="python", encoding='GBK')
    return data

class GPT4Rec:
    def __init__(self, user_data_path:str, movies_data_path:str,num_recommendations=5, beam_size=5) -> None:
        self.user_rating = get_user_data(user_data_path)
        self.movies = get_movies_data(movies_data_path)
        self.client = OpenAI(base_url="https://open.bigmodel.cn/api/paas/v4/",
                             api_key="be4dd5e44b4f438b954bef2e17de1037.kgtglAo6UCqxg0NF")
        self.beam_size = beam_size
        self.num_recommendations = num_recommendations

    def recommend_movies(self, user_id:int, num:int=10) :
        # 1. 获取用户最近打分高的电影列表
        user_movies = self.get_movie_by_user(user_id)
        if len(user_movies) == 0:
            return None

        history_movies = [f"标题:{title}, 类型:{movie_class}" for title, movie_class in zip(user_movies['title'], user_movies['class'])]

         # 2. 提示词构建
        prompt = (
            f"""用户最近看过的电影是：
{"\n".join(history_movies)}, 
接下来，用户可能会搜索："""
        )
        print(prompt)
        # 3. 大模型生成beam_size个查询
        response = self.client.chat.completions.create(
            model="glm-4.6",
            messages=[
                {"role": "system",
                 "content": f"你是一个专业的电影推荐助手，可以根据用户的观看历史预测他们接下来可能搜索什么。只输出一个编号列表,数量是{self.beam_size}，包含简短、真实的搜索查询（每行一个）。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.9,
            extra_body={
                "thinking": {
                    "type": "disabled",
                }, # 关闭思考
            }
        )
        text = response.choices[0].message.content

        queries = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # 移除序号前缀（如 "1. ", "2) ", "- "）
            clean_line = re.sub(r"^[0-9]+[.)]?\s*|-+\s*", "", line)
            if clean_line and len(clean_line) > 2:
                queries.append(clean_line)
            if len(queries) >= num:
                break

        print(f"=======Ai生成的查询=======：\n{queries}")
        # 4. bm25算法在电影库中和查询做检索
        all_candidates = []
        all_movies = self.movies['title'].drop_duplicates().tolist()
        for query in queries:
            candidates = self.bm25_retrival(query, all_movies, top_k=5)
            all_candidates.extend(candidates)

        # 去重并保留顺序
        seen = set()
        final_recommendations = []
        for item in all_candidates:
            if item not in seen:
                seen.add(item)
                final_recommendations.append(item)
            if len(final_recommendations) >= self.num_recommendations:
                break

        return final_recommendations

    def bm25_retrival(self, query:str, docs:List[str], top_k:int) -> List[str]:
        tokenized_doc= [doc.lower().split(" ") for doc in docs]
        bm25 = BM25Okapi(tokenized_doc)
        tokenized_query = query.lower().split(" ")
        top_docs = bm25.get_top_n(tokenized_query, docs, n=top_k)
        return top_docs

    def get_movie_by_user(self,user_id,num:int=10)-> pd.DataFrame:
        """
        获取用户打分高的指定数量的电影
        """
        user_movies = self.user_rating[(self.user_rating["user_id"] == user_id) & (self.user_rating["rating"] >= 3)].drop_duplicates()['movie_id']
        user_movies = self.movies[self.movies["id"].isin(user_movies)]
        if len(user_movies) < num:
            return user_movies
        else:
            return user_movies[0:num]




if __name__ == "__main__":
    # data = get_user_data("./M_ML-100K/ratings.dat")
    # print(data[0:5])
    #
    # movie = get_movies_data("./M_ML-100K/movies.dat")
    # print(movie[0:5])

    rec = GPT4Rec("./M_ML-100K/ratings.dat", "./M_ML-100K/movies.dat")
    movies = rec.recommend_movies(user_id=196)

    print(f"=======电影推荐=======：")
    if movies is None:
        print("用户没有看过任何电影")
    else:
        for movie in movies:
            print(f"- {movie}")

