### 数据管理接口
1. 文档上传接口
- post 上传pdf文件（图片/文档）
- {
- "file": List[file], 文件
- collection: str, 所属文档集合
- user: str, 上传账号
- createdAt: str 上传时间
- }
2. 文档查询
- get 查询文档
- {
    "collection": str,     # 可选，按集合过滤
    "page": int = 1,       # 分页
    "page_size": int = 20,
    "keyword": str         # 可选，关键词搜索
}
3. 文档删除
- delete 删除文档
- {
    "document_id": str,    # 路径参数
}

### 多模态检索接口
1. 多模态检索接口
- post 扩模态检索文本和图像内容
- {
    "query": str,                   # 查询文本
    "query_image": str = None,      # 可选，Base64编码的查询图像
    "collection": str = None,       # 可选，指定文档集合
    "top_k": int = 10               # 返回结果数量
}
2. 混合检索接口
- post  结合关键词和向量检索
- {
    "query": str
}

### 多模态问答接口
1. 基于多模态知识库额度问答
- post  基于多模态知识库的问答
- {
    "question": str,                # 用户问题
    "query_image": str = None,      # 可选，相关图像
    "collection": str = None,       # 指定文档集合
}
2. 问答历史接口
- get  获取问答历史
{
    "session_id": str = None,   # 会话ID
    "limit": int = 50,          # 返回数量
}