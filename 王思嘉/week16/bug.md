# 代码存在bug

## 1.索引越界问题
在call函数中，设置了k=100,但是如果数据总量达不到100条，便会发生索引越界；


## 2.字符串解码问题
在return self.redis.mget([self.name + "key:"+ q.decode() for q in filtered_prompts])中，未检查 q 是否已经是字符串类型就调用，如果 q 已经是字符串，会抛出 AttributeError 异常

## 3.缓存清除问题
在clear_cache函数中，它试图删除列表中存储的值而不是真正的键，无法正确清除 Redis 中的实际缓存项

## 4. 查询逻辑不一致

在call函数中，判断是否有匹配项的方式与实际使用的匹配项不一致，可能会跳过一些有效的匹配项或者在无匹配时仍然执行后续操作

修改：
```
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

        embedding = self.embedding_method(prompt)
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
        embedding = self.embedding_method(prompt)

        # 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=100)
        
        # 过滤不满足距离的结果
        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]
        
        # 如果没有满足条件的结果，返回None
        if not filtered_ind:
            return None

        prompts = self.redis.lrange(self.name + "list", 0, -1)
        # 确保索引不会越界
        filtered_prompts = [prompts[i] for i in filtered_ind if i < len(prompts)]

        # 获取得到原始的提问，并在redis找到对应的回答
        # 处理字节串解码问题
        keys = []
        for q in filtered_prompts:
            if isinstance(q, bytes):
                keys.append(self.name + "key:" + q.decode())
            else:
                keys.append(self.name + "key:" + q)
        
        return self.redis.mget(keys)

    def clear_cache(self):
        # 获取所有存储的提示词
        prompts = self.redis.lrange(self.name + "list", 0, -1)
        # 构造正确的键名列表用于删除
        keys_to_delete = []
        for prompt in prompts:
            if isinstance(prompt, bytes):
                keys_to_delete.append(self.name + "key:" + prompt.decode())
            else:
                keys_to_delete.append(self.name + "key:" + prompt)
        
        # 删除所有相关的键
        if keys_to_delete:
            self.redis.delete(*keys_to_delete)
        
        # 删除列表本身
        self.redis.delete(self.name + "list")
        
        # 删除索引文件
        if os.path.exists(f"{self.name}.index"):
            os.unlink(f"{self.name}.index")
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
```
