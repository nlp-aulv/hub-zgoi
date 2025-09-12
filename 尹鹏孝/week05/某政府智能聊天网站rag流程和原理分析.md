# 1、首先环境配置
## 1）使用conda环境在uv下运行。
## 2）主要包括下载bge-reranker-base和bge-small-zh-v1.5
## 3）用ollama安装qwen3:0.6b
## 4）安装es数据库
## 5）安装sqlLite或者mysql数据库
## 6）配置项目中的config.yml文件为本地的环境。特别是models模块下的embedding_model和 rerank_model分别对应
assets/models/hub/models/Xorbits/bge-small-zh-v1___5和./assets/models/hub/models/BAAI/bge-reranker-base/本地目录
配置rag大模型token和地址（可以是在线的大模型也可以是本地的ollama环境下运行的大模型qwen3:0.6b）：
  llm_base: "http://localhost:11434/v1"
  llm_api_key: "111"
  llm_model: "qwen3:0.6b"
## 7）device: "cuda"还需要更新pytorch为比较高的版本否则运行不起来。

# 2、rag在项目中的基本配置
## 1)构建一个RAG（Retrieval-Augmented Generation）系统，其中：
document_meta 存储文档的元信息
chunk_info 存储文档分块内容和对应的向量嵌入，用于语义检索
先导入配置并读取es向量数据库配置：
import yaml  # type: ignore
from elasticsearch import Elasticsearch  #type: ignore
import traceback
# 通过
with open('config.yml') as f:
     config = yaml.safe_load(f)#读取配置
     提取连接参数：
es_host = config["elasticsearch"]["host"]
es_port = config["elasticsearch"]["port"]
es_scheme = config["elasticsearch"]["scheme"]
es_username = config["elasticsearch"]["username"]
es_password = config["elasticsearch"]["password"]

## 2)建立连接：
if es_username != "" and es_password != "":
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
        basic_auth=(es_username, es_password)
    )
else:
    es = Elasticsearch(
        [{"host": es_host, "port": es_port, "scheme": es_scheme}],
    )

获取嵌入模型：
embedding_dims = config["models"]["embedding_model"][
    config["rag"]["embedding_model"]
]["dims"]

检查连接：
if not es.ping():
    print("Could not connect to Elasticsearch.")
    return False

## 3)定义document_meta_mapping映射
document_meta_mapping = {
    "mappings":{
        'properties': {
            'file_name': {
                'type': 'text',
                'analyzer': 'ik_max_word',  #使用IK中文分词器
                'search_analyzer': 'ik_max_word'
            },
            'abstract': {
                'type': 'text',
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_max_word'
            },
            'full_content': {
                'type': 'text',
                'analyzer': 'ik_max_word',
                'search_analyzer': 'ik_max_word'
            }
        }
    }
}

## 4)这个索引用于存储文档的元数据，所有文本字段都配置了IK中文分词器。

配置索引：
if not es.indices.exists(index="document_meta"):
    es.indices.create(index='document_meta', body=document_meta_mapping)
定义 chunk_info 索引映射
chunk_info_mapping = {
    'mappings': {
        'properties': {
            'chunk_content': {
                'type': 'text',                #字段类型为文本
                'analyzer': 'ik_max_word',     #索引时使用的分词器
                'search_analyzer': 'ik_max_word'  #搜索时使用的分词器
            },
            "embedding_vector": {
                "type": "dense_vector",  #密集向量类型
                "element_type": "float",
                "dims": embedding_dims,  #使用配置中的维度
                "index": True,  #启用索引
                "index_options": {
                    "type": "int8_hnsw"  #使用HNSW算法进行近似最近邻搜索
                }
            }
        }
    }
}
这个索引用于存储文本块及其对应的向量嵌入，支持向量相似度搜索。

## 5)还定义了两个mysql或者sqLite数据表knowledge_database、knowledge_document分别用于存储文档分类和知识分类


# 3、在RAG中使用过程和原理：
## 1）、首先加载模型和文档知识数据和历史数据->

## 2）对文档进行了chunk切割核心代码是对pdf做了切割：对word做了跳过。在对pdf切割时候做了图片提取

## 3）使用EMBEDDING_MODEL_PARAMS对知识进程重排，使用了bge-small-zh-v1.5和bge-base-zh-v1.5来实现文本转固定向量和理解文本和文本，文本和句子的相似度，和匹配度。
bge-base-zh-v1.5准确度更高， 所以速度低，bge-small-zh-v1.5速度高但是准确率低，bge-base-zh-v1.5适合高质量检索。bge-small-zh-v1.5适合及时响应，提升了用户检索体验、
使用了bge-reranker-base进行排序。
## 4）对文本进行编码：get_embedding
## 5）在全局检索中： 指定一个知识库检索，bm25打分，选择出：初步计算得到top 100的待选文档，  fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
筛选最相关的50个，
倒数排名融合（RRF）使得：排名越前，分数越高，但差异逐渐减小。多路召回融合基于字面匹配，召回精确匹配的文档，基于语义相似度，召回语义相关的文档，这里分别有关键词法，向量法，热门词，
新屋召回等多路召回通过组合多种方法的优势，弥补单一方法的不足，从而显著提高召回率（Recall），确保尽可能多的相关知识不被漏掉。
for idx, record in enumerate(word_search_response['hits']['hits']):使用了关键词召回。
for idx, record in enumerate(vector_search_response['hits']['hits']):使用了向量召回，
召回后使用：倒数排名融合（RRF - Reciprocal Rank Fusion）为每个文件在不同召回列表中的排名赋予一个分数，排名越靠前，分数越高。
然后将同一个知识在不同列表中的分数相加，得到总分，最后按总分重新排序：RRF Score = 1 / (rank + k)，然后使用了if self.use_rerank:
    rerank_score = self.rerank(text_pair) # 使用精细模型重排序，让数据更加准确。
## 6）对提问过来的数据如果是第一次则从文档中检索:chat_with_rag
message #对话历史消息列表,
如果message是1则是首次对话先从RAG的方式
使用前面的多路召回系统
在指定知识库中查找与用户问题相关的文档片段
构建RAG动态提示模板
rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
    .replace("{#QUESTION#}", query) \
    .replace("{#RELATED_DOCUMENT#}", related_document)
调用大语言模型生成回答
rag_response = self.chat(
    [{"role": "user", "content": rag_query}],  #构建好的提示
    0.7, 0.9  #温度参数和top-p参数
).content
取数据，后续对话则对于多轮对话，不进行检索
直接基于对话上下文生成回复
然后更新并回复
messages.append({"role": "system", "content": rag_response})
return messages








