# RAG的实现流程（以政企问答项目代码为例）

**从代码流程里面来开，RAG其实就是一种提示词工程，他将用户提问最为相关的资料信息以提示词的形式，作为system prompt输入到模型中，达到基于已有知识问答的效果**

目前看来，实现RAG的上下游流程如下：

1. 数据收集与处理
2. 检索（Retrieval-Augmented）与生成（Generation）
3. 如何为RAG补充知识信息？（待学习）



## 1. 背景（为什么需要RAG？）

LLM如同一个刚毕业的高中生，虽然知识深度不足，但是知识广度远超上了几年大学生的大学生，简直就是上知天文下通地理。

但同时带来的局限是，如果我们想要LLM去像一个律师一样对法律条文有深入的研究，针对现象引经据典解决问题是不太可能做到的。

这是因为LLM在generate的时候可能会出现幻觉现象，包括但不限于：和法学生一样考试的时候临时”立法“，使用根本不存在的法条...

那为了"缓解" （而不是也无法完全消除）大模型的幻觉现象，让大模型有可靠的知识来源生成回答，这就需要为大模型输入数据。

这一过程就像是大学期末考试前，为学生划重点，这样即便是大脑空空的大学生也知道在重点范围内找考试答案。

那么，为大学生（LLM）划重点考试的过程就是RAG的实现过程，我们首先需要：

１.　限定一门课程并划重点（数据收集与处理）

２.　学生根据重点答题（检索与生成）

３.　如果教材更新了，老师怎么为下一批学生补充重点信息





##  2. 数据收集与处理

### 2.1 数据收集

数据收集一般来说有两种渠道：

a. 开源数据：互联网上可获取且没有版权纠纷的数据（有版权纠纷的数据只要没有被发现，那知识产权保护法就没有作用【狗头】）

b. 非开源数据：企业内部或者客户提供的非开源数据。



### 2.2 数据处理

对于1.1中获取的所有数据，无论是pdf， 图片， 表格还是其他的花花绿绿的格式，数据处理的唯一目的永远是：

a. 将各类数据向量化，变成计算机可以识别的格式

b. 保存到数据库中，方便检索

#### 2.2.1 文本分块

文本分块对应项目代码的中**extract_content()**方法

从代码实现流程来说:

a. pdf文件按页读取并向量化之后保存到数据库

b. 页数据按chunk size分解成一个一个的chunk，然后将chunk向量化之后保存到数据库

```
RAG().extract_content()
```

```
def _extract_pdf_content(self, knowledge_id, document_id, title, file_path) -> bool:
    try:
        pdf = pdfplumber.open(file_path)
    except:
        print("打开文件失败")
        return False

    print(f"{file_path} pages: ", len(pdf.pages))  # 打印提示信息，显示PDF文件的页数

    abstract = ""

    for page_number in range(len(pdf.pages)): # 每一页 提取
        current_page_text = pdf.pages[page_number].extract_text() # 提取图片
        if page_number <=3:
            abstract = abstract + '\n' + current_page_text

        # 每一页内容的内容
        embedding_vector = self.get_embedding(current_page_text)
        page_data = {
            "document_id": document_id,
            "knowledge_id": knowledge_id,
            "page_number": page_number,
            "chunk_id": 0, # 先存储每一也所有内容
            "chunk_content": current_page_text,
            "chunk_images": [],
            "chunk_tables": [],
            "embedding_vector": embedding_vector
        }
        response = es.index(index="chunk_info", document=page_data)

        # 划分chunk
        page_chunks = split_text_with_overlap(current_page_text, self.chunk_size, self.chunk_overlap)
        embedding_vector = self.get_embedding(page_chunks)
        for chunk_idx in range(1, len(page_chunks) + 1):
            page_data = {
                "document_id": document_id,
                "knowledge_id": knowledge_id,
                "page_number": page_number,
                "chunk_id": chunk_idx,
                "chunk_content": page_chunks[chunk_idx - 1],
                "chunk_images": [],
                "chunk_tables": [],
                "embedding_vector": embedding_vector[chunk_idx - 1]
            }
            response = es.index(index="chunk_info", document=page_data)

    document_data = {
        "document_id": document_id,
        "knowledge_id": knowledge_id,
        "document_name": title,
        "file_path": file_path,
        "abstract": abstract
    }
    response = es.index(index="document_meta", document=document_data)
```



#### 2.2.2 数据向量化

数据向量化的方式直接决定了检索的准确率。

数据向量化的手段是词嵌入（Embedding）

词嵌入手段很多，包括但不限于：Bert模型， 大模型官方的embedding接口

对应代码中的方法如下：

```
def get_embedding(self, text) -> np.ndarray:
    """
    对文本进行编码
    :param text: 待编码文本
    :return: 编码结果
    """
    if self.embedding_model in ["bge-small-zh-v1.5", "bge-base-zh-v1.5"]:
        return EMBEDDING_MODEL_PARAMS["embedding_model"].encode(text, normalize_embeddings=True)

    raise NotImplemented
```





## 3. 检索与生成

当我们需要让大模型回答相关领域问题时，内部处理逻辑是：

a. 提取用户的问题的重点

b. 针对重点检索相关知识块

c. 将知识嵌入到提示词中喂给大模型

d. 大模型根据知识生成对应回答



#### 3.1 根据用户提问检索信息

这个过程就类似搜索引擎，我们可以使用**TF-IDF**或者其改进算法**BM25**去数据库中检索相关的知识块信息。

但是，现在都2025年，谁还在用传统检索办法。

所以，为了考虑到语义信息，我们可以对用户的提问进行**文本向量化**，然后在向量数据库中使用余弦相似度获取相关的知识块

那么，这两种检索办法办法哪种比较好呢？

答案是：我全都要

在我们这些未来的AI算法工程师严重，"我全都要"这种表述叫做**"多路召回"**，根据两种办法综合起来，可以减少重要信息被筛选漏的概率。

同时，对于**海**出来的这些知识块，我们需要重新打分排序，这样根据排序结果，我们就可以获取到与用户提问最为相关的知识信息。

在代码中的对应方法：

```
def query_document(self, query: str, knowledge_id: int) -> List[str]:
    # 全文检索，指定一个知识库检索，bm25打分
    word_search_response = es.search(index="chunk_info", 
        body={
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                "chunk_content": query
                            }
                        }
                    ],
                    "filter": [
                        {
                            "term": {
                                "knowledge_id": knowledge_id
                            }
                        }
                    ]
                }
            },
            "size": 50
        },
        fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
        source=False,
    )

    # 语义检索
    embedding_vector = self.get_embedding(query) # 编码
    knn_query = {
        "field": "embedding_vector",
        "query_vector": embedding_vector,
        "k": 50,
        "num_candidates": 100, # 初步计算得到top 100的待选文档， 筛选最相关的50个
        "filter": {
            "term": {
                "knowledge_id": knowledge_id
            }
        }
    }
    vector_search_response = es.search(
        index="chunk_info", knn=knn_query,
        fields=["chunk_id", "document_id", "knowledge_id", "page_number", "chunk_content"],
        source=False,
    )

    # rrf
    # 检索1 ：[a， b， c]
    # 检索2 ：[b， e， a]
    # a 1/60    b 1/61    c 1/62
    # b 1/60    e 1/61    a 1/62

    k = 60
    fusion_score = {}
    search_id2record = {}
    for idx, record in enumerate(word_search_response['hits']['hits']):    
        _id = record["_id"]
        if _id not in fusion_score:
            fusion_score[_id] = 1 / (idx + k)
        else:
            fusion_score[_id] += 1 / (idx + k)

        if _id not in search_id2record:
            search_id2record[_id] = record["fields"]

    for idx, record in enumerate(vector_search_response['hits']['hits']):    
        _id = record["_id"]
        if _id not in fusion_score:
            fusion_score[_id] = 1 / (idx + k)
        else:
            fusion_score[_id] += 1 / (idx + k)

        if _id not in search_id2record:
            search_id2record[_id] = record["fields"]

    sorted_dict = sorted(fusion_score.items(), key=lambda item: item[1], reverse=True)
    sorted_records = [search_id2record[x[0]] for x in sorted_dict][:self.chunk_candidate]
    sorted_content = [x["chunk_content"] for x in sorted_records]

    if self.use_rerank:
        text_pair = []
        for chunk_content in sorted_content:
            text_pair.append([query, chunk_content])
        rerank_score = self.rerank(text_pair) # 重排序打分
        rerank_idx = np.argsort(rerank_score)[::-1]

        sorted_records = [sorted_records[x] for x in sorted_records]
        sorted_content = [sorted_content[x] for x in sorted_content]

    return sorted_records
```

```
# chunk打分
def get_rank(self, text_pair) -> np.ndarray:
    """
    对文本对进行重排序
    :param text_pair: 待排序文本
    :return: 匹配打分结果
    """
    if self.rerank_model in ["bge-reranker-base"]:
        with torch.no_grad():
            inputs = EMBEDDING_MODEL_PARAMS["rerank_tokenizer"](
                text_pair, padding=True, truncation=True,
                return_tensors='pt', max_length=512,
            )
            inputs = {key: value.to(device) for key, value in inputs.items()}
            scores = EMBEDDING_MODEL_PARAMS["rerank_model"](**inputs, return_dict=True).logits.view(-1, ).float()
            scores = scores.data.cpu().numpy()
            return scores

    raise NotImplemented
```



#### 3.2 大模型生成回答

我们通过3.1以及获取相关的知识片段了，接下来为了让大模型能够读到这些信息，这些信息就被输入到大模型之中

在提示中，有一类提示词叫做**system prompt**，他是大模型cosplay的关键

我们首先设定好相关prompt模板，将3.1的知识块替换进去之后，

大模型就会按照我们的要求，扮演指定的角色，根据提供的知识，结合自己优秀的口才，生成相关的回答。

这也就是为什么有些人会认为RAG没有知识含量，因为RAG本质就是一种提示词工程，

他的技术难点以及重点都在于如何将各类数据向量化，以及如何以一种更好的方式检索出来。

对于大模型能不能根据知识合理回答，这些都是大模型本身的能力限制，我们无法干预。

对应代码如下：

```
BASIC_QA_TEMPLATE = '''现在的时间是{#TIME#}。你是一个专家，你擅长回答用户提问，帮我结合给定的资料，回答下面的问题。
    如果问题无法从资料中获得，或无法从资料中进行回答，请回答无法回答。如果提问不符合逻辑，请回答无法回答。
    如果问题可以从资料中获得，则请逐步回答。

    资料：
    {#RELATED_DOCUMENT#}


    问题：{#QUESTION#}
    '''
    
def chat_with_rag(
    self,
    knowledge_id: int, # 知识库 哪一个知识库提问
    messages: List[Dict],
):
    if len(messages) == 1:
        query = messages[0]["content"]
        related_records = self.query_document(query, knowledge_id) # 检索到相关的文档
        print(related_records)
        related_document = '\n'.join([x["chunk_content"][0] for x in related_records])

        rag_query = BASIC_QA_TEMPLATE.replace("{#TIME#}", str(datetime.datetime.now())) \
            .replace("{#QUESTION#}", query) \
            .replace("{#RELATED_DOCUMENT#}", related_document)

        rag_response = self.chat(
            [{"role": "user", "content": rag_query}],
            0.7, 0.9
        ).content
        messages.append({"role": "system", "content": rag_response})
    else:
        normal_response = self.chat(
            messages,
            0.7, 0.9
        ).content
        messages.append({"role": "system", "content": normal_response})

    # messages.append({"role": "system", "content": rag_response})
    return messages
```



## 4. 如何补充数据

不会，要学，要长脑子了

