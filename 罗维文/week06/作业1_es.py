from elasticsearch import Elasticsearch
import json

def print_search_results(response):
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档内容：{json.dumps(hit['_source'], ensure_ascii=False, indent=2)}")

es = Elasticsearch("http://localhost:9200")

# 检查连接
if es.ping():
    print("成功连接到 Elasticsearch！")
else:
    print("无法连接到 Elasticsearch，请检查服务是否运行。")

index_name = "products"

# 检查索引是否存在，如果不存在则创建
if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "product_id": {"type": "keyword"},
                    "name": {"type": "text", "analyzer": "ik_max_word"},
                    "description": {"type": "text", "analyzer": "ik_smart"},
                    "price": {"type": "float"},
                    "category": {"type": "keyword"},
                    "stock": {"type": "integer"},
                    "on_sale": {"type": "boolean"},
                    "created_at": {"type": "date"}
                }
            }
        }
    )
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")

# 插入一个新文档
doc_1 = {
    "product_id": "A001",
    "name": "智能手机",
    "description": "最新款智能手机，性能强大，拍照清晰。",
    "price": 4999.50,
    "category": "电子产品",
    "stock": 150,
    "on_sale": True,
    "created_at": "2023-01-15T09:00:00Z"
}
es.index(index=index_name, id="A001", document=doc_1)
print("文档 'A001' 已插入。")

# 插入另一个文档
doc_2 = {
    "product_id": "B002",
    "name": "无线蓝牙耳机",
    "description": "音质卓越，佩戴舒适，超长续航。",
    "price": 699.00,
    "category": "电子产品",
    "stock": 300,
    "on_sale": True,
    "created_at": "2023-02-20T14:30:00Z"
}
es.index(index=index_name, id="B002", document=doc_2)
print("文档 'B002' 已插入。")

# 插入第三个文档
doc_3 = {
    "product_id": "C003",
    "name": "智能手表",
    "description": "支持心率监测、运动追踪和消息提醒，防水设计。",
    "price": 1299.99,
    "category": "电子产品",
    "stock": 80,
    "on_sale": True,
    "created_at": "2023-03-10T11:15:00Z"
}
es.index(index=index_name, id="C003", document=doc_3)
print("文档 'C003' 已插入。")

# 刷新索引以确保文档可被搜索到
es.indices.refresh(index=index_name)

# 1. 全文检索（使用 'match' 查询）
# 搜索名称或描述中包含“智能”的商品
print("\n--- 检索 1: 全文检索“智能” ---")
res_1 = es.search(
    index=index_name,
    body={
        "query": {
            "multi_match": {
                "query": "智能",
                "fields": ["name", "description"]
            }
        }
    }
)
print_search_results(res_1)


# 2. 结合 'filter' 进行精确过滤
# 搜索价格低于 1000 元且正在促销的电子产品
print("\n--- 检索 2: 结合查询与过滤 ---")
res_2 = es.search(
    index=index_name,
    body={
        "query": {
            "bool": {
                "must": {
                    "match_all": {}  # 匹配所有文档
                },
                "filter": [
                    {"term": {"category": "电子产品"}},
                    {"term": {"on_sale": True}},
                    {"range": {"price": {"lt": 1000}}}
                ]
            }
        }
    }
)
print_search_results(res_2)


# 3. 按关键词分组聚合
# 统计不同类别的商品数量
print("\n--- 检索 3: 聚合查询（按类别统计） ---")
res_3 = es.search(
    index=index_name,
    body={
        "aggs": {
            "products_by_category": {
                "terms": {
                    "field": "category",
                    "size": 10
                }
            }
        },
        "size": 0  # 不返回文档结果，只返回聚合结果
    }
)
print(json.dumps(res_3['aggregations']['products_by_category']['buckets'], ensure_ascii=False, indent=2))

# 检索：电子产品中价格低于 2000 元的商品
print("\n--- 电子产品中价格低于 2000 元的商品 ---")
res_4 = es.search(
    index=index_name,
    body={
        "query": {
            "bool": {
                "must": [
                    {"term": {"category": "电子产品"}},
                    {"range": {"price": {"lt": 2000.00}}}
                ]
            }
        }
    }
)

print_search_results(res_4)