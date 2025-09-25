from elasticsearch import Elasticsearch
from datetime import datetime

# 连接到 Elasticsearch
es = Elasticsearch("http://localhost:9200")

# 检查连接
if es.ping():
    print("成功连接到 Elasticsearch！")
else:
    print("无法连接到 Elasticsearch，请检查服务是否运行。")
    exit()

# 定义索引名称
index_name = "users"

# 创建索引（如果不存在）
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "user_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "ik_max_word"},
                    "email": {"type": "keyword"},
                    "age": {"type": "integer"},
                    "city": {"type": "keyword"},
                    "created_at": {"type": "date"}
                }
            }
        }
    )
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")

# 1. 增加文档 (Create)
print("\n--- 1. 增加文档 ---")
user_data = {
    "user_id": "U001",
    "name": "张三",
    "email": "zhangsan@example.com",
    "age": 25,
    "city": "北京",
    "created_at": datetime.now()
}

# 插入文档，指定ID为U001
result = es.index(index=index_name, id="U001", document=user_data)
print(f"文档插入成功，ID: {result['_id']}")

# 插入第二个文档
user_data2 = {
    "user_id": "U002",
    "name": "李四",
    "email": "lisi@example.com",
    "age": 30,
    "city": "上海",
    "created_at": datetime.now()
}
es.index(index=index_name, id="U002", document=user_data2)
print(f"文档插入成功，ID: U002")

# 刷新索引确保文档可搜索
es.indices.refresh(index=index_name)

# 2. 查询文档 (Read)
print("\n--- 2. 查询文档 ---")

# 查询所有文档
print("所有用户:")
result = es.search(index=index_name, body={"query": {"match_all": {}}})
for hit in result['hits']['hits']:
    print(f"ID: {hit['_id']}, 姓名: {hit['_source']['name']}, 年龄: {hit['_source']['age']}")

# 根据ID查询特定文档
print("\n查询特定用户(U001):")
result = es.get(index=index_name, id="U001")
print(f"用户详情: {result['_source']}")

# 条件查询 - 查找年龄大于28的用户
print("\n年龄大于28的用户:")
query = {
    "query": {
        "range": {
            "age": {"gt": 28}
        }
    }
}
result = es.search(index=index_name, body=query)
for hit in result['hits']['hits']:
    print(f"姓名: {hit['_source']['name']}, 年龄: {hit['_source']['age']}")

# 3. 更新文档 (Update)
print("\n--- 3. 更新文档 ---")
# 更新U001的年龄
update_data = {
    "doc": {
        "age": 26,
        "city": "广州"
    }
}
result = es.update(index=index_name, id="U001", body=update_data)
print(f"文档更新成功: {result['result']}")

# 验证更新
result = es.get(index=index_name, id="U001")
print(f"更新后的用户详情: {result['_source']}")

# 4. 删除文档 (Delete)
print("\n--- 4. 删除文档 ---")
# 删除U002文档
result = es.delete(index=index_name, id="U002")
print(f"文档删除成功: {result['result']}")

# 验证删除后的文档列表
print("\n删除后的用户列表:")
result = es.search(index=index_name, body={"query": {"match_all": {}}})
for hit in result['hits']['hits']:
    print(f"ID: {hit['_id']}, 姓名: {hit['_source']['name']}")

print(f"剩余用户数量: {result['hits']['total']['value']}")

# 关闭连接
es.close()
print("\n操作完成，连接已关闭。")
