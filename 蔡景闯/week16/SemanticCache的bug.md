# 1. 结果获取错误

```python
# 向量数据库中进行检索
        dis, ind = self.index.search(embedding, k=100)
        if dis[0][0] > self.distance_threshold:
            return None

        # 过滤不满足距离的结果
        filtered_ind = [i for i, d in enumerate(dis[0]) if d < self.distance_threshold]

        pormpts = self.redis.lrange(self.name + "list", 0, -1)
        print("pormpts", pormpts)
        filtered_prompts = [pormpts[i] for i in filtered_ind]
```

代码中的`filtered_prompts = [pormpts[i] for i in filtered_ind]`错误，因为 **Faiss 的索引顺序 ≠ Redis list 的索引顺序**

修改方案：在 Faiss 添加向量时，**记录每个向量 ID 到 Redis**

```python
def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
    if isinstance(prompt, str):
        prompt = [prompt]
        response = [response]
  
    embedding = self.embedding_method(prompt)
    dim = embedding.shape[1]
  
    if self.index is None:
        # 使用 IDMap 保存 ID -> 向量
        self.index = faiss.IndexIDMap(faiss.IndexFlatL2(dim))
        next_id = 0
    else:
        # 从 Redis 获取当前最大 ID（或用 incr）
        next_id = int(self.redis.get(self.name + ":next_id") or 0)

    with self.redis.pipeline() as pipe:
        ids = []
        for i, (q, a) in enumerate(zip(prompt, response)):
            # 用唯一 ID 作为 key 的一部分
            key = f"{self.name}:key:{next_id + i}"
            pipe.setex(key, self.ttl, a)
            ids.append(next_id + i)
      
        pipe.set(self.name + ":next_id", next_id + len(prompt))
        pipe.execute()

    # 添加带 ID 的向量
    ids_array = np.array(ids).astype('int64')
    self.index.add_with_ids(embedding, ids_array)
    faiss.write_index(self.index, f"{self.name}.index")

def call(self, prompt: str):
    if self.index is None:
        return None
    embedding = self.embedding_method([prompt])
    dis, ids = self.index.search(embedding, k=10)
  
    valid_ids = [int(id_) for d, id_ in zip(dis[0], ids[0]) if d < self.distance_threshold and id_ != -1]
    if not valid_ids:
        return None
      
    keys = [f"{self.name}:key:{id_}" for id_ in valid_ids]
    return self.redis.mget(keys)
```

# 2. `clear_cache()` 删除逻辑错误

## 2.1 删除键错误

```python
pormpts = self.redis.lrange(self.name + "list", 0, -1)
self.redis.delete(*pormpts)
```

保存的时候使用的key格式是：`self.name + "key:" + q`，但是删除使用的是保存进去列表里面的查询`q`，应该改成：

```python
keys_to_delete = [self.name + "key:" + q.decode() for q in prompts]
```

## 2.2 删除时超出Redis的数量限制

删除时报错：`wrong number of arguments for 'del' command`

应该修改成分批次删除：

```python
batch_size = 100
    for i in range(0, len(keys_to_delete), batch_size):
        batch = keys_to_delete[i:i + batch_size]
        self.redis.delete(*batch)
```
