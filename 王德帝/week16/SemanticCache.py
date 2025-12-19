import os
import numpy as np
import redis
from typing import Optional, List, Union, Callable, Any
import faiss

class SemanticCache:
    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int=3600*24, # 过期时间
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold=0.1
    ):
        self.name = name
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password
        )
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.embedding_method = embedding_method

        if os.path.exists(f"{self.name}.index"):
            self.index = faiss.read_index(f"{self.name}.index")
        else:
            self.index = None

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        # BUG说明：原来这里直接使用 embedding_method 的返回结果，如果返回的是 float64 或一维向量，
        # 会导致 faiss.IndexFlatL2.add 报错（FAISS 要求 float32 且为二维数组）。
        # 修复方法：统一转成 numpy.float32，并保证是二维矩阵。
        embedding = self.embedding_method(prompt)
        embedding = np.array(embedding, dtype="float32")  # 修复：强制转换为 float32
        if embedding.ndim == 1:  # 修复：保证是二维 (n, d)
            embedding = embedding.reshape(1, -1)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        self.index.add(embedding)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(self.name + "key:" + q, self.ttl, a) # 提问和回答存储在redis
                    pipe.lpush(self.name + "list", q) #所有的提问都存储在list里面，方便后续使用

                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1

    def call(self, prompt: str):
        if self.index is None:
            return None

        # 新的提问进行编码
        # BUG说明：原来这里直接传入字符串 prompt，如果 embedding 方法只支持 List[str]，
        # 或返回一维向量，会导致后续 self.index.search 形状不符合要求。
        # 修复方法：与 store 保持一致，先转 List[str]，并统一成 float32 的二维矩阵。
        embedding = self.embedding_method([prompt])  # 修复：统一使用列表接口
        embedding = np.array(embedding, dtype="float32")  # 修复：强制转换为 float32
        if embedding.ndim == 1:  # 修复：保证是二维 (1, d)
            embedding = embedding.reshape(1, -1)

        # 向量数据库中进行检索
        # BUG说明：当索引中的向量数量少于 100 时，直接使用 k=100 会导致 faiss 抛出错误。
        # 修复方法：k 不能超过 self.index.ntotal。
        k = min(100, self.index.ntotal)  # 修复：限制最大检索数量
        dis, ind = self.index.search(embedding, k=k)
        if dis[0][0] > self.distance_threshold:
            return None

        # 过滤不满足距离的结果
        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        pormpts = self.redis.lrange(self.name + "list", 0, -1)
        print("pormpts", pormpts)
        filtered_prompts = [pormpts[i] for i in filtered_ind]

        # 获取得到原始的提问 ，并在redis 找到对应的回答
        return self.redis.mget([self.name + "key:"+ q.decode() for q in filtered_prompts])

    def clear_cache(self):
        pormpts = self.redis.lrange(self.name + "list", 0, -1)
        # BUG说明：原来这里直接 delete(*pormpts)，但 pormpts 只是“用户提问文本”，
        # 实际写入 redis 的 key 是 self.name + "key:" + q，导致 value 永远不会真正删除。
        # 修复方法：根据 pormpts 还原出真正的 key，再删除这些 key。
        if pormpts:
            keys = [self.name + "key:" + q.decode() for q in pormpts]  # 修复：构造正确的缓存 key
            self.redis.delete(*keys)
            self.redis.delete(self.name + "list")

        # BUG说明：如果 index 文件不存在，直接 unlink 会抛 FileNotFoundError。
        # 修复方法：先判断文件是否存在。
        index_path = f"{self.name}.index"
        if os.path.exists(index_path):  # 修复：安全删除索引文件
            os.unlink(index_path)
        self.index = None

if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])


    embed_cache = SemanticCache(
        name="semantic_ache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
    )

    embed_cache.clear_cache()

    embed_cache.store(prompt="hello world", response="hello world1232")
    print(embed_cache.call(prompt="hello world"))

    embed_cache.store(prompt="hello my bame", response="nihao")
    print(embed_cache.call(prompt="hello world"))