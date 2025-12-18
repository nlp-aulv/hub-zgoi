import redis

r = redis.Redis(
    host="127.0.0.1",
    port=6379,
    db=0,
    decode_responses=True  # 返回 str 而不是 bytes
)

# 基本测试
r.set("hello", "world")
print(r.get("hello"))  # world

# ================= 列表 List =================

# 先清空一下
r.delete("mylist")

r.lpush("mylist", "a", "b")      # 列表: ['b', 'a']
r.rpush("mylist", "c")           # 列表: ['b', 'a', 'c']
print("mylist 全部元素:", r.lrange("mylist", 0, -1))

print("lpop mylist:", r.lpop("mylist"))   # b
print("rpop mylist:", r.rpop("mylist"))   # c
print("mylist 剩余元素:", r.lrange("mylist", 0, -1))
