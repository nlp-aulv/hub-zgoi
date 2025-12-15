import json

from elasticsearch import Elasticsearch, exceptions

# 本机Elasticsearch配置信息
ES_HOST = "https://localhost:9200"
ES_USER = "elastic"
ES_PASSWORD = "SW7abdbiVWGmzOkM7tRQ"

# 创建 ES 客户端，禁用证书验证（仅本地测试用）
es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False
)
def print_search_results(response):
    print(f"找到 {response['hits']['total']['value']} 条文档：")
    for hit in response['hits']['hits']:
        print(f"得分：{hit['_score']}，文档内容：{json.dumps(hit['_source'], ensure_ascii=False, indent=2)}")


def check_connection():
    try:
        info = es.info()
        print("ES 连接成功：", info['name'])
    except exceptions.ConnectionError as e:
        print("ES 连接失败，请检查服务是否启动。", e)
        return False
    return True

# def list_analyzer():
#     try:
#         response = es.indices.analyze(body={"text": "测试"}, explain=True)
#         print("默认分析器可用")
#     except exceptions.BadRequestError:
#         print("未指定分析器时无法分析文本")
#
#
#     for analyzer in ["ik_smart", "ik_max_word"]:
#         try:
#             res = es.indices.analyze(body={"analyzer": analyzer, "text": "我讨厌上班，我讨厌上这个不挣钱的B班。"})
#             tokens = [t['token'] for t in res['tokens']]
#             print(f"{analyzer} 分词结果:", tokens)
#         except exceptions.BadRequestError as e:
#             print(f"{analyzer} 分词请求失败:", e.info['error']['reason'])


# 检查连接
if es.ping():
    print("连接成功！")
else:
    print("连接失败！")

index_name = "financial-transaction-logs-2025.12"

if not es.indices.exists(index=index_name):
    es.indices.create(
        index=index_name,
        body={
              "mappings": {
                "properties": {
                  "transaction_id": {
                    "type": "keyword"
                  },
                  "transaction_time": {
                    "type": "date"
                  },

                  "account": {
                    "properties": {
                      "account_id": { "type": "keyword" },
                      "customer_id": { "type": "keyword" },
                      "account_type": { "type": "keyword" }
                    }
                  },

                  "amount": {
                    "properties": {
                      "value": { "type": "double" },
                      "currency": { "type": "keyword" }
                    }
                  },

                  "channel": {
                    "type": "keyword"
                  },

                    "merchant": {
                        "properties": {
                            "name": {
                                "type": "text",
                                "analyzer": "ik_max_word",
                                "fields": {
                                    "keyword": {
                                        "type": "keyword"
                                    }
                                }
                            }
                        }
                    },


                  "location": {
                    "properties": {
                      "country": { "type": "keyword" },
                      "city": { "type": "keyword" },
                      "ip": { "type": "ip" }
                    }
                  },

                  "device": {
                    "properties": {
                      "device_id": { "type": "keyword" },
                      "os": { "type": "keyword" },
                      "app_version": { "type": "keyword" }
                    }
                  },

                  "risk": {
                    "properties": {
                      "risk_score": { "type": "integer" },
                      "risk_level": { "type": "keyword" },
                      "hit_rules": { "type": "keyword" }
                    }
                  },

                  "result": {
                    "properties": {
                      "status": { "type": "keyword" },
                      "reason": { "type": "keyword" }
                    }
                  }
                }
              }
            }
    )
    print(f"索引 '{index_name}' 创建成功。")
else:
    print(f"索引 '{index_name}' 已存在。")


doc_1 = {
  "transaction_id": "TXN202512150089",
  "transaction_time": "2025-12-15T08:47:12Z",

  "account": {
    "account_id": "ACC_5729018",
    "customer_id": "CUST_330845",
    "account_type": "credit"
  },

  "amount": {
    "value": 12999.00,
    "currency": "CNY"
  },

  "channel": "web",
  "merchant": {
    "name": "境外数码产品商城",
  },

  "location": {
    "country": "SG",
    "city": "Singapore",
    "ip": "203.116.32.18"
  },

  "device": {
    "device_id": "DEV_f9c23e7a",
    "os": "Windows",
    "app_version": "web_1.0"
  },

  "risk": {
    "risk_score": 85,
    "risk_level": "high",
    "hit_rules": ["R003", "R021", "R089"]
  },

  "result": {
    "status": "rejected",
    "reason": "cross_border_high_risk"
  }
}
es.index(index=index_name, id="TXN202512150089", document=doc_1)
print("文档 'A001' 已插入。")

doc_2 = {
  "transaction_id": "TXN202512150102",
  "transaction_time": "2025-12-15T09:15:36Z",

  "account": {
    "account_id": "ACC_5729018",
    "customer_id": "CUST_330845",
    "account_type": "debit"
  },

  "amount": {
    "value": 268.50,
    "currency": "CNY"
      },
  "merchant": {
    "properties": {
       "name": {
          "type": "text",
          "analyzer": "ik_max_word",
          "fields": {
            "keyword": {
              "type": "keyword"
            }
          }
        }
      }
    },

  "channel": "mobile_app",
  "merchant": {
    "name": "社区生鲜超市",
  },

  "location": {
    "country": "CN",
    "city": "Shanghai",
    "ip": "101.89.23.77"
  },

  "device": {
    "device_id": "DEV_f9c23e7a",
    "os": "Android",
    "app_version": "8.3.0"
  },

  "risk": {
    "risk_score": 8,
    "risk_level": "low",
    "hit_rules": []
  },

  "result": {
    "status": "approved",
    "reason": "normal_transaction"
  }
}
es.index(index=index_name, id="TXN202512150102",document=doc_2)
print("文档 'TXN202512150102' 已插入。")

es.indices.refresh(index=index_name)

print("\n--- 检索 1:低风险 + 交易成功 ---")
res_1 = es.search(
    index=index_name,
    body={
          "query": {
            "bool": {
              "must": [
                { "term": { "risk.risk_level": "low" } },
                { "term": { "result.status": "approved" } }
              ]
            }
          }
        }
)
print_search_results(res_1)

# print("\n--- 检索2:超市 ---")
# res_2 = es.search(
#     index=index_name,
#     body={
#         "query": {
#             "match": {
#                 "merchant.name": "超市"
#             }
#         }
#     }
# )
print("\n--- 检索：超市（多字段）---")
res_2 = es.search(
    index=index_name,
    body={
        "query": {
            "multi_match": {
                "query": "超市",
                "fields": [
                    "merchant.name",
                    "remark"
                ]
            }
        }
    }
)

print_search_results(res_2)

print("\n--- 检索：超市 ---")
res_3 = es.search(
    index=index_name,
    body={
        "query": {
            "match": {
                "merchant.name": "超市"
            }
        }
    }
)
print_search_results(res_3)
