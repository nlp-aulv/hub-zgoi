# pip install elasticsearch
from elasticsearch import Elasticsearch

# 替换为你的 Elasticsearch 地址
ELASTICSEARCH_URL = "http://localhost:9200"

# 如果没有安全认证，直接创建客户端
es_client = Elasticsearch(ELASTICSEARCH_URL)

# 测试连接
if es_client.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")

# 定义索引名称和映射
index_name = "blog_posts_py"
mapping = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "title": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "content": {
                "type": "text",
                "analyzer": "ik_max_word",
                "search_analyzer": "ik_smart"
            },
            "tags": {"type": "keyword"},
            "author": {"type": "keyword"},
            "created_at": {"type": "date"}
        }
    }
}

# 检查索引是否存在，如果不存在则创建
if not es_client.indices.exists(index=index_name):
    es_client.indices.create(index=index_name, body=mapping)
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已经存在。")

from datetime import datetime

documents = [
    {
        "title": "Elasticsearch 入门指南",
        "content": "这是一篇关于如何安装和使用 Elasticsearch 的详细文章。学习搜索技术可以提升数据处理能力。",
        "tags": ["Elasticsearch", "教程", "搜索"],
        "author": "张三",
        "created_at": datetime(2023, 10, 26, 10, 0, 0)
    },
    {
        "title": "深入理解IK分词器",
        "content": "IK分词器是中文分词的优秀工具。它的智能分词和最细粒度分词模式各有优势。",
        "tags": ["分词", "IK", "中文"],
        "author": "李四",
        "created_at": datetime(2023, 10, 25, 15, 30, 0)
    }
]

for doc in documents:
    es_client.index(index=index_name, document=doc)
    print(f"文档已插入: '{doc['title']}'")

# 刷新索引，确保文档可被搜索到
es_client.indices.refresh(index=index_name)


# 定义查询函数
def search_docs(query):
    response = es_client.search(index=index_name, body=query)
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档：{hit['_source']['title']}")


# 1. 查询标题中的 "入门指南"
print("\n--- 1. 查询标题中的 '入门指南' ---")
query_1 = {
    "query": {
        "match": {
            "title": "入门指南"
        }
    }
}
search_docs(query_1)

# 2. 结合全文和精确匹配查询
print("\n--- 2. 结合全文（搜索技术）和精确匹配（作者：张三） ---")
query_2 = {
    "query": {
        "bool": {
            "must": {
                "match": {
                    "content": "搜索技术"
                }
            },
            "filter": {
                "term": {
                    "author": "张三"
                }
            }
        }
    }
}
search_docs(query_2)
