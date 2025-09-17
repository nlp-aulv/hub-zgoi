# pip install elasticsearch
from elasticsearch import Elasticsearch, NotFoundError
from datetime import datetime

# 替换为你的 Elasticsearch 地址
ELASTICSEARCH_URL = "http://localhost:9200"

# 如果没有安全认证，直接创建客户端
es_client = Elasticsearch(ELASTICSEARCH_URL)

# 测试连接
if es_client.ping():
    print("连接成功！")
else:
    print("连接失败。请检查 Elasticsearch 服务是否运行。")
    raise 1

# 创建索引
index_name = "es_index_test" # 索引：相当于表名
mapping = {
  "settings": {
    "number_of_shards": 1, # 索引被拆分的份数
    "number_of_replicas": 0 # 副本
  },
  "mappings": { # 映射：相当于字段定义 schema
    "properties": {
      "title": { # 字段：title、content、tags、author、created_at
        "type": "text", # 字段类型
        "analyzer": "ik_max_word", # 分析器
        "search_analyzer": "ik_smart"
      },
      "content": {
        "type": "text",
        "analyzer": "ik_max_word",
        "search_analyzer": "ik_smart"
      },
      "tags": { "type": "keyword" },
      "author": { "type": "keyword" },
      "created_at": { "type": "date" },
      "views": { "type": "integer" },
      "rating": { "type": "float" }
    }
  }
}

es_client.indices.create(index=index_name, body=mapping)
print(f"索引 '{index_name}' 创建成功。")


documents = [
  {
    "title": "RAG技术全面解析与应用指南",
    "content": "RAG（Retrieval-Augmented Generation）结合了检索和生成的优势，通过外部知识库增强大语言模型的能力。本文详细介绍了RAG的工作原理和实际应用场景。",
    "tags": ["RAG", "人工智能", "检索增强生成", "LLM"],
    "author": "李四",
    "created_at": "2023-11-08T09:30:00",
    "views": 1560,
    "rating": 4.8
  },
  {
    "title": "如何构建高效的RAG系统：从入门到精通",
    "content": "学习如何使用Elasticsearch作为向量数据库构建RAG系统，实现准确的信息检索和高质量的文本生成。本文包含详细的代码示例和最佳实践。",
    "tags": ["RAG系统", "Elasticsearch", "向量搜索", "AI应用"],
    "author": "王五",
    "created_at": "2023-12-01T14:20:00",
    "views": 2890,
    "rating": 4.9
  },
  {
    "title": "RAG与传统搜索技术的对比分析",
    "content": "本文比较了RAG技术与传统关键词搜索的差异，探讨了RAG在语义理解、上下文感知和生成质量方面的优势。",
    "tags": ["RAG", "搜索技术", "语义搜索", "对比分析"],
    "author": "赵六",
    "created_at": "2023-11-20T16:45:00",
    "views": 1230,
    "rating": 4.6
  },
  {
    "title": "使用Python实现RAG管道的完整教程",
    "content": "手把手教你使用LangChain、Elasticsearch和OpenAI API构建完整的RAG应用程序。包含环境配置、数据索引和查询处理的全流程。",
    "tags": ["RAG", "Python", "LangChain", "实战教程"],
    "author": "张三",
    "created_at": "2023-12-05T11:15:00",
    "views": 3450,
    "rating": 4.7
  },
  {
    "title": "RAG在企业知识管理中的应用实践",
    "content": "探讨RAG技术如何帮助企业构建智能知识库系统，提升员工工作效率和决策质量。案例研究显示RAG可以显著减少信息检索时间。",
    "tags": ["RAG应用", "企业知识管理", "AI助手", "数字化转型"],
    "author": "李四",
    "created_at": "2023-11-25T13:10:00",
    "views": 980,
    "rating": 4.5
  },
  {
    "title": "优化RAG系统性能的10个技巧",
    "content": "本文分享了提升RAG系统响应速度、准确性和稳定性的实用技巧，包括索引优化、查询重写和缓存策略等。",
    "tags": ["RAG优化", "性能调优", "搜索算法", "技术技巧"],
    "author": "王五",
    "created_at": "2023-12-10T10:30:00",
    "views": 2100,
    "rating": 4.8
  },
  {
    "title": "RAG在医疗领域的创新应用",
    "content": "研究RAG技术如何辅助医疗诊断、文献检索和患者咨询，提高医疗服务的准确性和效率。",
    "tags": ["RAG医疗", "人工智能医疗", "智能诊断", "健康科技"],
    "author": "赵六",
    "created_at": "2023-11-27T15:20:00",
    "views": 1670,
    "rating": 4.9
  },
  {
    "title": "基于Elasticsearch的RAG向量搜索实现",
    "content": "详细介绍如何使用Elasticsearch的kNN搜索功能构建RAG系统的检索模块，包括向量化、索引设计和相似度计算。",
    "tags": ["Elasticsearch", "向量搜索", "kNN", "RAG实现"],
    "author": "张三",
    "created_at": "2023-12-08T09:45:00",
    "views": 2980,
    "rating": 4.7
  },
  {
    "title": "RAG技术面临的挑战与未来发展趋势",
    "content": "分析当前RAG系统在准确性、实时性和可解释性方面的挑战，并展望未来的技术发展方向和应用前景。",
    "tags": ["RAG挑战", "技术趋势", "AI发展", "未来展望"],
    "author": "李四",
    "created_at": "2023-12-12T14:00:00",
    "views": 1350,
    "rating": 4.6
  },
  {
    "title": "多模态RAG：结合文本、图像和音频的检索生成",
    "content": "探索下一代RAG系统如何整合多种模态信息，实现更丰富、更准确的检索和生成能力。",
    "tags": ["多模态RAG", "多媒体检索", "AI生成", "技术创新"],
    "author": "王五",
    "created_at": "2023-12-15T11:30:00",
    "views": 1890,
    "rating": 4.8
  }
]

for doc in documents:
    es_client.index(index=index_name, document=doc)
    print(f"文档已插入: '{doc['title']}'")

# 刷新索引，确保文档可被搜索到
es_client.indices.refresh(index=index_name)

def search_docs(query):
    response = es_client.search(index=index_name, body=query)
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档：{hit['_source']['title']}")

# 1. 查询标题中的 "入门指南"
print("\n--- 1. 查询标题中的 'RAG技术' ---")
query_1 = {
  "query": {
    "match": {
      "title": "RAG技术"
    }
  }
}
search_docs(query_1)

# 2. 查询评分高于4.7分的文档
print("\n--- 2. 查询评分高于4.8分的文档 ---")
query_2 = {
  "query": {
    "range":{
        "rating": {
            "gt": 4.7
        }
    }
  }
}
search_docs(query_2)

# 3. 查询指定时间范围的的文档
print("\n--- 3. 查询指定时间范围的的文档 ---")
query_2 = {
  "query": {
    "range":{
        "created_at": {
            "gte": "2023-11-30T15:20:00",
            "lte": "2023-12-15T11:30:00"
        }
    }
  }
}
search_docs(query_2)


def delete_index(index_name):
    """
    删除指定的ES索引
    """
    try:
        # 检查索引是否存在
        if es_client.indices.exists(index=index_name):
            # 删除索引
            response = es_client.indices.delete(index=index_name)
            print(f"索引 '{index_name}' 删除成功")
            print(f"响应: {response}")
            return True
        else:
            print(f"索引 '{index_name}' 不存在")
            return False

    except NotFoundError:
        print(f"索引 '{index_name}' 不存在")
        return False
    except Exception as e:
        print(f"删除索引时发生错误: {e}")
        return False


# 使用示例
print('------删除索引--------')
delete_index(f"{index_name}")