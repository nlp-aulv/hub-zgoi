from typing import Optional, List, Union, Any, Dict, Callable
import redis
import os
import faiss
import json


class SemanticRouter:
    def __init__(
        self,
        name: str,
        embedding_method: Callable[[Union[str, List[str]]], Any],
        ttl: int = 3600 * 24,
        redis_url: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        distance_threshold=0.3
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
        self.routes = {}

        # 初始化向量索引
        self.index_file = f"{self.name}_routes.index"
        if os.path.exists(self.index_file):
            self.index = faiss.read_index(self.index_file)
            # 加载路由配置
            routes_data = self.redis.get(f"{self.name}_routes_config")
            if routes_data:
                self.routes = json.loads(routes_data)
        else:
            self.index = None

    def add_route(self, questions: List[str], target: str):
        if target not in self.routes:
            self.routes[target] = {
                'questions': [],
                'embeddings': None
            }

        # 添加问题到路由
        for q in questions:
            if q not in self.routes[target]['questions']:
                self.routes[target]['questions'].append(q)

        # 为问题生成嵌入向量
        embeddings = self.embedding_method(self.routes[target]['questions'])
        self.routes[target]['embeddings'] = embeddings

        # 保存配置和索引
        self.index.add(embeddings)
        self.redis.setex(
            f"{self.name}_routes_config",
            self.ttl,
            json.dumps(self.routes)
        )
        faiss.write_index(self.index, self.index_file)

        # 为每个问题创建缓存键
        for q in questions:
            self.redis.setex(
                f"{self.name}_route_cache:{q}",
                self.ttl,
                target
            )

    def route(self, question: str):
        # 检查缓存
        cached_result = self.redis.get(f"{self.name}_cache:{question}")
        if cached_result:
            return cached_result.decode()

        # 生成问题嵌入向量
        embedding = self.embedding_method([question])

        # 搜索最相似的参考问题
        dis, ind = self.index.search(embedding, k=10)

        best_target = None
        best_distance = float('inf')

        # 寻找最佳匹配
        for i, (distance, idx) in enumerate(zip(dis[0], ind[0])):
            if distance < best_distance:
                # 找到对应的路由目标
                for target, route_data in self.routes.items():
                    if idx < len(route_data['questions']):
                        best_target = target
                        best_distance = distance
                        break

        # 缓存结果
        if best_target and best_distance < self.distance_threshold:
            self.redis.setex(
                f"{self.name}_cache:{question}",
                self.ttl,
                best_target
            )
            return best_target

        return None

if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    router("Hi, good morning")