import redis
from redis.exceptions import ConnectionError, TimeoutError

def test_redis_operations():
    try:
        # 1. 连接 Redis（默认本地 6379 端口，无密码）
        r = redis.Redis(
            host='localhost',
            port=6379,
            db=0,               # 使用数据库 0
            decode_responses=True  # 自动将 bytes 解码为 str（方便打印）
        )

        # 测试连接
        r.ping()
        print("✅ 成功连接到 Redis 服务器！")

        # ==============================
        # 2. 字符串（String）操作
        # ==============================
        print("\n--- 字符串（String）操作 ---")
        r.set("name", "Alice")
        r.setex("temp_token", 10, "abc123")  # 设置带过期时间（10秒）
        print("name:", r.get("name"))
        print("temp_token (10秒内有效):", r.get("temp_token"))

        # ==============================
        # 3. 哈希（Hash）操作
        # ==============================
        print("\n--- 哈希（Hash）操作 ---")
        user_key = "user:1001"
        r.hset(user_key, mapping={"name": "Bob", "age": "30", "email": "bob@example.com"})
        print("用户信息:", r.hgetall(user_key))
        print("用户年龄:", r.hget(user_key, "age"))

        # ==============================
        # 4. 列表（List）操作
        # ==============================
        print("\n--- 列表（List）操作 ---")
        task_list = "tasks"
        r.rpush(task_list, "task1", "task2", "task3")  # 从右边插入
        print("任务列表:", r.lrange(task_list, 0, -1))  # 获取全部
        popped = r.lpop(task_list)  # 从左边弹出一个
        print("已处理任务:", popped)
        print("剩余任务:", r.lrange(task_list, 0, -1))

        # ==============================
        # 5. 集合（Set）操作
        # ==============================
        print("\n--- 集合（Set）操作 ---")
        tags1 = "post:1:tags"
        tags2 = "post:2:tags"
        r.sadd(tags1, "python", "redis", "database")
        r.sadd(tags2, "redis", "cache", "performance")
        print("文章1标签:", r.smembers(tags1))
        print("两篇文章共同标签:", r.sinter(tags1, tags2))  # 交集
        print("所有唯一标签:", r.sunion(tags1, tags2))      # 并集

        # ==============================
        # 6. 有序集合（Sorted Set）操作
        # ==============================
        print("\n--- 有序集合（Sorted Set）操作 ---")
        leaderboard = "game:scores"
        r.zadd(leaderboard, {"Alice": 100, "Bob": 150, "Charlie": 120})
        print("排行榜（按分数升序）:", r.zrange(leaderboard, 0, -1, withscores=True))
        print("Top 2 高分玩家:", r.zrevrange(leaderboard, 0, 1, withscores=True))  # 降序

        # ==============================
        # 7. 键操作 & 删除
        # ==============================
        print("\n--- 清理测试数据 ---")
        keys_to_delete = ["name", "temp_token", user_key, task_list, tags1, tags2, leaderboard]
        deleted_count = r.delete(*keys_to_delete)
        print(f"✅ 已删除 {deleted_count} 个测试键。")

    except ConnectionError:
        print("❌ 无法连接到 Redis，请确保 Redis 服务正在运行（localhost:6379）")
    except TimeoutError:
        print("❌ 连接 Redis 超时")
    except Exception as e:
        print(f"❌ 发生错误: {e}")

def explore_keys(prefix="*", count=20):
    """安全扫描并展示指定前缀的 key 及其简要信息"""
    print(f"🔍 正在扫描匹配 '{prefix}' 的 key（最多 {count} 个）...")
    r = redis.Redis(
        host='localhost',
        port=6379,
        db=0,  # 使用数据库 0
        decode_responses=True  # 自动将 bytes 解码为 str（方便打印）
    )
    cursor = 0
    found = 0
    while found < count:
        cursor, keys = r.scan(cursor=cursor, match=prefix, count=10)
        for key in keys:
            if found >= count:
                break
            key_type = r.type(key)
            if key_type == "string":
                value = r.get(key)
                preview = str(value)[:50]  # 截断长字符串
            elif key_type == "hash":
                value = r.hgetall(key)
                preview = f"Hash({len(value)} fields)"
            elif key_type == "list":
                length = r.llen(key)
                preview = f"List({length} items)"
            elif key_type == "set":
                size = r.scard(key)
                preview = f"Set({size} members)"
            elif key_type == "zset":
                size = r.zcard(key)
                preview = f"ZSet({size} members)"
            else:
                preview = f"<{key_type}>"

            print(f"🔑 {key:<30} | {key_type:<6} | {preview}")
            found += 1
        if cursor == 0 or found >= count:
            break

if __name__ == "__main__":
    test_redis_operations()
    explore_keys("user:*")  # 查看所有用户