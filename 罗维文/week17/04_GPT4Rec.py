"""
GPT4Rec ：基于生成式语言模型的个性化推荐框架，借助 GPT2 生成未来的查询条件，使用搜索检索到相关的物品。
- 步骤1（生成查询条件）: 根据用户历史交互物品的文本信息（如商品标题），生成能够代表用户未来兴趣的、可读的“搜索查询”。
    Previously, the customer has bought: <标题1>. <标题2>... In the future, the customer wants to buy
- 步骤2（物品的检索）: 从整个物品库中检索出最相关的物品作为推荐候选
"""
import pandas as pd
import jieba
from openai import OpenAI
from rank_bm25 import BM25Okapi
ratings = pd.read_csv("../../week17-Multimodal_Datasets/M_ML-100K/ratings.dat", sep="::", header=None, engine='python')
ratings.columns = ["user_id", "movie_id", "rating", "timestamp"]


movies = pd.read_csv("../../week17-Multimodal_Datasets/M_ML-100K/movies.dat", sep="::", header=None, engine='python', encoding="latin")
movies.columns = ["movie_id", "movie_title", "movie_tag"]

movies_title_tag = ["movie_title:" + str(i) + "\tmovie_tag:" + str(j) for i, j in zip(movies['movie_title'], movies['movie_tag'])]

userid = 716  # 即将推荐的用户id
# 获取该用户评分（大于等于3分）最高的前10个 movie_id
top_movie_ids = ratings[(ratings['user_id'] == userid) & (ratings['rating'] >= 3)].sort_values('rating', ascending=False).head(10)['movie_id']
# 根据 movie_id 从 movies 表中提取对应的 movie_title 和 movie_tag
top_movie = movies[movies['movie_id'].isin(top_movie_ids)]

# 构造字符串列表
user_movies = ["movie_title:" + str(i) + "\tmovie_tag:" + str(j) for i, j in zip(top_movie['movie_title'], top_movie['movie_tag'])]


PROMPT_TEMPLATE = """
你是一个电影推荐专家，请结合用户历史观看的电影，推荐用户未来可能观看的电影，每一行是一个推荐的电影名字：

如下是历史观看的电影：
{0}

请基于上述电影进行推荐，推荐10个待选的电影描述，每一行是一个推荐。
返回格式
```推荐列表
movie_title:推荐1 movie_tag:推荐1
movie_title:推荐2 movie_tag:推荐2
...
movie_title:推荐10 movie_tag:推荐10
```
""".format('\n'.join(user_movies))

print(PROMPT_TEMPLATE)

client = OpenAI(api_key="", base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

# Round 1
messages = [{"role": "user", "content": PROMPT_TEMPLATE}]
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages
)

content = response.choices[0].message.content

# print(content)


if '```推荐列表' in content:
    user_movie_title_tag = content.split('```推荐列表')[1].split('```')[0].strip()

    # bm25 打分
    movies_words = [jieba.lcut(x) for x in movies_title_tag]
    bm25 = BM25Okapi(movies_words)
    # 每个推荐文本与电影进行打分
    for movie in user_movie_title_tag.split('\n'):
        movie_scores = bm25.get_scores(jieba.lcut(movie))
        # 最匹配的前五个电影的下标
        max_score_movie_idx = movie_scores.argsort()[::-1][:5]
        print('推荐查询文本:', movie)
        print('查询到的对应电影:')
        print([str(movies.loc[i, 'movie_id']) + movies_title_tag[i] for i in max_score_movie_idx])
        print('查询到的对应电影分数:')
        print([movie_scores[i] for i in max_score_movie_idx])
        print('-' * 50)


