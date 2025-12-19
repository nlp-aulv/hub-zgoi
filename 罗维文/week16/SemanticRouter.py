from typing import Optional, List, Union, Any, Dict, Callable
import os
import numpy as np
import redis
import faiss
class SemanticRouter:
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


    def add_route(self, questions: List[str], target: str):
        embeddings = self.embedding_method(questions)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(embeddings)
        faiss.write_index(self.index, f"{self.name}.index")

        try:
            with self.redis.pipeline() as pipe:
                for q in questions:
                    pipe.setex(self.name + "key:" + q, self.ttl, target) # 提问和标签存储在redis
                    pipe.rpush(self.name + "list", q) #所有的提问都存储在list里面，方便后续使用

                return pipe.execute()
        except:
            import traceback
            traceback.print_exc()
            return -1


    def route(self, question: str):
        if self.index is None:
            return None

        # 新的提问进行编码
        embedding = self.embedding_method(question)

        # 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=1)
        if dis[0][0] > self.distance_threshold:
            return None

        questions = self.redis.lrange(self.name + "list", 0, -1)
        print("questions", questions)
        filtered_questions = questions[ind[0][0]].decode()

        # 获取得到原始的提问 ，并在redis 找到对应的标签
        return self.redis.get(self.name + "key:"+ filtered_questions)

if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]

        return np.array([np.ones(768) for t in text])

    router = SemanticRouter(        
        name="semantic_router",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        )

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    router("Hi, good morning")