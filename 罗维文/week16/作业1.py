import redis

r = redis.Redis(host='localhost', port=6379, decode_responses=True)

print("=== String类型 ===")

# 设置键值
r.set('name', '李武')
r.set('age', '20')
r.setex('temp_key', 60, '临时数据')

name = r.get('name')
age = r.get('age')
temp_key = r.get('temp_key')
print(name, age, temp_key)  # 李武 20 临时数据

r.delete('name')
r.delete('age')
r.delete('temp_key')

r.setnx('cnt', 0)
r.incr('cnt')
r.incrby('cnt', 5)
r.decr('cnt')
r.decrby('cnt', 5)

cnt = r.get('cnt')
print(cnt)  # 0

r.delete('cnt')

r.mset({'cnt': cnt, 'name':name, 'age': age, 'temp_key': temp_key})
value = r.mget('cnt', 'name', 'age','temp_key')
print(value)  # ['0', '李武', '20', '临时数据']
# 清空当前数据库（例如 db0）的所有 key
# r.flushdb()
# 清空所有数据库
# r.flushall()
# 异步清空当前库
# r.flushdb(asynchronous=True)
# 异步清空所有库
# r.flushall(asynchronous=True)
print("清空前 key 数量:", r.dbsize())  # 清空前 key 数量: 4
r.flushdb()
print("清空后 key 数量:", r.dbsize())  # 清空后 key 数量: 0

print("\n=== List类型 ===")

r.lpush('mylist', '第一个')
r.rpush('mylist', '第二个')
r.lpush('mylist', '第零个')

all_items = r.lrange('mylist', 0, -1)
print(all_items)  # ['第零个', '第一个', '第二个']

# 弹出元素
left_item = r.lpop('mylist')
right_item = r.rpop('mylist')
print(f"左侧弹出: {left_item}, 右侧弹出: {right_item}")  # 左侧弹出: 第零个, 右侧弹出: 第二个

all_items = r.lrange('mylist', 0, -1)
print(all_items)  # ['第一个']

length = r.llen('mylist')
print(f"列表长度: {length}")  # 列表长度: 1

print("清空前 key 数量:", r.dbsize())  # 清空前 key 数量: 1
r.flushdb()
print("清空后 key 数量:", r.dbsize())  # 清空后 key 数量: 0

print("\n=== Set类型 ===")

r.sadd('tags1', 'python', 'redis', 'cache')
r.sadd('tags2', 'python', 'cache', 'java')

tags = r.smembers('tags1')
print(type(tags), tags)  # <class 'set'> {'redis', 'cache', 'python'}

intersection = r.sinter('tags1', 'tags2')  # 交集
union = r.sunion('tags1', 'tags2')  # 并集
diff = r.sdiff('tags1', 'tags2')  # 差集

print(f"交集: {intersection}")  # 交集: {'cache', 'python'}
print(f"并集: {union}")  # 并集: {'redis', 'cache', 'java', 'python'}
print(f"差集: {diff}")  # 差集: {'redis'}

# 随机弹出
random_tag = r.spop('tags1')
print(f"随机弹出: {random_tag}")  # 随机弹出: redis

tags = r.smembers('tags1')
print(tags)  # {'cache', 'python'}

print("清空前 key 数量:", r.dbsize())  # 清空前 key 数量: 2
r.flushdb()
print("清空后 key 数量:", r.dbsize())  # 清空后 key 数量: 0

print("\n=== ZSet有序集合 ===")

# 添加带分数的成员
r.zadd('rank', {
    'Alice': 95,
    'Bob': 87,
    'Charlie': 92,
    'David': 78
})

top3 = r.zrevrange('rank', 0, 2, withscores=True)# withscores在返回成员（members）的同时，也返回它们对应的分数（scores）
print(f"前三名: {top3}")  # 前三名: [('Alice', 95.0), ('Charlie', 92.0), ('Bob', 87.0)]

# 获取分数范围
good_students = r.zrangebyscore('rank', 90, 100, withscores=True)
print(f"90分以上: {good_students}")  # 90分以上: [('Charlie', 92.0), ('Alice', 95.0)]

# 获取排名
bob_rank = r.zrevrank('rank', 'Bob')
bob_score = r.zscore('rank', 'Bob')
print(f"Bob排名: {bob_rank}, 分数: {bob_score}")  # Bob排名: 2, 分数: 87.0

print("清空前 key 数量:", r.dbsize())  # 清空前 key 数量: 1
r.flushdb()
print("清空后 key 数量:", r.dbsize())  # 清空后 key 数量: 0

