

---

### **第一阶段：知识库构建**

将原始文档（如PDF文件）处理并存入向量数据库（Elasticsearch），为后续的检索做准备。

1.  **模型加载**：
    *   在脚本初始化时，根据`config.yaml`的配置，加载指定的**嵌入模型**用于将文本转换为向量。
    *   加载**重排序模型**（如 `bge-reranker-base`）用于对初步检索结果进行精细化排序。

2.  **文档处理 (`extract_content` 和 `_extract_pdf_content`)**：
    *   用户或系统调用 `extract_content` 方法，传入文档ID、类型、文件路径等。
    *   对于PDF文件，调用 `_extract_pdf_content`：
        *   使用 `pdfplumber` 逐页读取文本内容。
        *   **生成两种粒度的文本块 (Chunk)**：
            *   **Page-level Chunk**：将整页内容作为一个块，并计算其嵌入向量（`embedding_vector`），存入ES索引 `chunk_info`。
            *   **Sub-page Chunk**：使用 `split_text_with_overlap` 函数，将每页文本按配置的 `chunk_size` 和 `chunk_overlap` 进一步切分成更小的片段，分别计算嵌入向量，并存入 `chunk_info`。这有助于检索更精确的上下文。
        *   同时，将文档元信息（如标题、摘要——前3页内容拼接）存入 `document_meta` 索引。

3.  **数据存储**：
    *   所有处理后的文本块及其对应的嵌入向量、元数据都存储在Elasticsearch的 `chunk_info` 索引中。构成系统的“知识库”。

---

### **第二阶段：问答交互**

当用户发起提问时，系统通过 `chat_with_rag` 方法进行处理。

1.  **判断对话轮次**：
    *   代码中通过 `if len(messages) == 1:` 判断是否为用户的**首轮提问**。如果是，则触发RAG流程；否则，可能将其视为多轮对话中的后续轮次，直接交由LLM处理（不进行检索）。

2.  **检索相关文档 (`query_document`)**：
    这是RAG的核心步骤，目标是从 `chunk_info` 索引中找出与用户问题最相关的文本片段。
    *   **步骤 2.1: 全文检索 (BM25)**：
        *   使用Elasticsearch的 `match` 查询，在指定 `knowledge_id` 的范围内，根据关键词匹配度（BM25算法）检索出Top 50个相关文本块。
    *   **步骤 2.2: 语义检索 (向量相似度)**：
        *   使用加载的嵌入模型，将用户问题 `query` 编码成向量 `embedding_vector`。
        *   在Elasticsearch中执行 `knn` 查询，在指定 `knowledge_id` 范围内，找出与问题向量最相似的Top 50个文本块（从初步筛选的100个候选中选出）。
    *   **步骤 2.3: 结果融合 (RRF - Reciprocal Rank Fusion)**：
        *   将上述两种检索方法的结果进行融合。RRF算法会给在两个列表中排名都靠前的文档更高的综合分数。
        *   根据融合分数对所有候选文档进行排序，取前 `chunk_candidate`（配置项）个作为初步结果。
    *   **步骤 2.4: 重排序 (Rerank)**：
        *   如果启用重排序 (`use_rerank`)，则将用户问题与上一步选出的每个候选文本块组成文本对 `[query, chunk_content]`。
        *   使用加载的重排序模型（如 `bge-reranker-base`）为每个文本对计算一个相关性得分。
        *   根据这个得分对候选文本块进行**重新排序**，得到最终的、质量更高的相关文档列表 `sorted_records`。

3.  **构造Prompt并调用LLM (`chat_with_rag`)**：
    *   将最终检索到的相关文本块内容 `chunk_content` 拼接成一个字符串 `related_document`。
    *   使用预定义的模板 `BASIC_QA_TEMPLATE`，将**当前时间**、**拼接的相关文档**和**用户原始问题**填充进去，构造一个结构化的Prompt。
    *   调用 `chat` 方法，将这个Prompt作为新的用户消息发送给配置的大语言模型（如通过OpenAI API）。
    *   LLM接收到包含“证据”的Prompt后，会基于这些信息生成答案。

4.  **返回答案**：
    *   将LLM生成的答案作为系统回复，添加到对话历史 `messages` 中，并返回给用户。

