# 多模态RAG项目接口设计（文本+图像分离）

## 1. 数据管理接口

| 接口名称 | 方法 | 路径 | 接口功能 | 关键入参字段 | 关键返回字段 |
|---------|------|------|----------|-------------|-------------|
| 纯文本文件上传 | POST | `/api/v1/files/text/upload` | 上传纯文本文件，支持PDF、Word、TXT等 | `file文本文件`, `title文档标题`, `description文档描述` | `file_id文件ID`, `text_chunk_count文本分块数` |
| 图像文件上传 | POST | `/api/v1/files/image/upload` | 上传图像文件，支持JPG、PNG等格式 | `file图像文件`, `image_title图像标题`, `image_description图像描述` | `file_id文件ID`, `image_embedding_status图像向量化状态` |
| 图文混合文件上传 | POST | `/api/v1/files/multimodal/upload` | 上传包含图文混合的文件，如PDF、PPT等 | `file混合文件`, `document_type文档类型`, `extract_both是否同时提取文本和图像` | `file_id文件ID`, `text_chunks文本块数`, `image_chunks图像块数` |
| 文件列表查询 | GET | `/api/v1/files` | 按类型查询文件列表 | `file_type文件类型(text/image/mixed)`, `page页码` | `files文件列表`, `type_count各类型数量` |
| 文件删除 | DELETE | `/api/v1/files/{file_id}` | 删除指定文件及所有相关索引 | `file_id文件ID` | `deleted_file_id已删除文件ID`, `cleaned_chunks清理块数` |

## 2. 多模态检索接口

| 接口名称 | 方法 | 路径 | 接口功能 | 关键入参字段 | 关键返回字段 |
|---------|------|------|----------|-------------|-------------|
| 纯文本检索 | POST | `/api/v1/search/text` | 基于文本内容的语义检索 | `text_query文本查询`, `top_k返回数量`, `filters过滤条件` | `text_results文本结果`, `text_scores相似度分数` |
| 图像检索 | POST | `/api/v1/search/image` | 基于图像内容的视觉检索 | `image_query图像查询(base64)`, `top_k返回数量`, `similarity_threshold相似度阈值` | `image_results图像结果`, `visual_scores视觉相似度` |
| 图文混合检索 | POST | `/api/v1/search/multimodal` | 文本+图像的联合检索 | `text_query文本查询`, `image_query图像查询`, `fusion_weight融合权重` | `multimodal_results多模态结果`, `combined_scores综合分数` |
| 文搜图检索 | POST | `/api/v1/search/text-to-image` | 用文本描述检索相关图像 | `text_query文本描述`, `top_k返回数量` | `image_results图像结果`, `cross_modal_scores跨模态分数` |
| 图搜文检索 | POST | `/api/v1/search/image-to-text` | 用图像检索相关文本内容 | `image_query图像查询`, `top_k返回数量` | `text_results文本结果`, `cross_modal_scores跨模态分数` |

## 3. 多模态问题接口

| 接口名称 | 方法 | 路径 | 接口功能 | 关键入参字段 | 关键返回字段 |
|---------|------|------|----------|-------------|-------------|
| 纯文本问答 | POST | `/api/v1/qa/text` | 基于纯文本内容的问答 | `text_question文本问题`, `context_files相关文件` | `text_answer文本答案`, `text_sources文本来源` |
| 图像问答 | POST | `/api/v1/qa/image` | 基于图像内容的视觉问答 | `image_question图像相关问题`, `target_images目标图像` | `visual_answer视觉答案`, `image_references图像引用` |
| 图文混合问答 | POST | `/api/v1/qa/multimodal` | 同时基于文本和图像的问答 | `multimodal_question多模态问题`, `text_context文本上下文`, `image_context图像上下文` | `multimodal_answer多模态答案`, `mixed_sources混合来源` |
| 图像描述生成 | POST | `/api/v1/qa/image-caption` | 为图像生成详细描述 | `target_image目标图像`, `detail_level详细程度` | `image_caption图像描述`, `key_elements关键元素` |
| 文档图解问答 | POST | `/api/v1/qa/document-visual` | 针对图文混合文档的问答 | `document_question文档问题`, `document_id文档ID` | `comprehensive_answer综合答案`, `text_citations文本引用`, `image_citations图像引用` |

## 4. 系统管理接口

| 接口名称 | 方法 | 路径 | 接口功能 | 关键入参字段 | 关键返回字段 |
|---------|------|------|----------|-------------|-------------|
| 系统状态检查 | GET | `/api/v1/system/status` | 检查文本和图像处理组件状态 | 无 | `text_processor_status文本处理器状态`, `image_processor_status图像处理器状态` |
| 索引统计 | GET | `/api/v1/system/index/stats` | 获取文本和图像索引统计 | `index_type索引类型` | `text_index_stats文本索引统计`, `image_index_stats图像索引统计` |
| 处理队列状态 | GET | `/api/v1/system/queue/status` | 查看文本和图像处理队列状态 | 无 | `text_queue文本队列状态`, `image_queue图像队列状态` |