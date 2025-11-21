# 多模态RAG系统 - 核心分层架构

## 核心数据流层级
```
用户层 → 知识库层 → 文档层 → 内容层 → 向量层
```

---

## 1. 用户层 (核心表)

### users 用户表
| 字段 | 类型 | 说明 |
|------|------|------|
| user_id | VARCHAR(64) | 用户唯一标识 |
| username | VARCHAR(128) | 用户名 |
| role | ENUM | 用户角色 |
| created_time | TIMESTAMP | 注册时间 |

---

## 2. 知识库层 (核心表)

### knowledge_bases 知识库表
| 字段 | 类型 | 说明 |
|------|------|------|
| kb_id | VARCHAR(64) | 知识库标识 |
| kb_name | VARCHAR(256) | 知识库名称 |
| owner_id | VARCHAR(64) | 拥有者ID |
| created_time | TIMESTAMP | 创建时间 |

---

## 3. 文档层 (核心表)

### documents 文档表
| 字段 | 类型 | 说明 |
|------|------|------|
| doc_id | VARCHAR(64) | 文档标识 |
| kb_id | VARCHAR(64) | 知识库ID |
| title | VARCHAR(512) | 文档标题 |
| source_type | ENUM | 文档类型 |
| file_path | VARCHAR(1024) | 文件路径 |
| created_time | TIMESTAMP | 创建时间 |

---

## 4. 内容层 (核心表)

### content_chunks 内容分块表
| 字段 | 类型 | 说明 |
|------|------|------|
| chunk_id | VARCHAR(64) | 分块标识 |
| doc_id | VARCHAR(64) | 文档ID |
| kb_id | VARCHAR(64) | 知识库ID |
| chunk_type | ENUM | 分块类型 |
| content_data | JSON | 内容数据(文本/图像/表格) |
| content_hash | VARCHAR(64) | 内容哈希 |

---

## 5. 向量层

### 5.1 向量元数据表 (关系数据库)
**vector_metadata**

| 字段 | 类型 | 说明 |
|------|------|------|
| vector_id | VARCHAR(64) | 向量记录ID |
| chunk_id | VARCHAR(64) | 分块ID |
| collection_name | VARCHAR(128) | Milvus集合名 |
| milvus_id | BIGINT | Milvus内部ID |

### 5.2 Milvus集合 (核心集合)

#### text_vectors 文本向量集合
| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(64) | 记录ID |
| chunk_id | VARCHAR(64) | 分块ID |
| text_vector | FLOAT_VECTOR | 文本向量(768维) |
| kb_id | VARCHAR(64) | 知识库ID |

#### image_vectors 图像向量集合
| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(64) | 记录ID |
| chunk_id | VARCHAR(64) | 分块ID |
| image_vector | FLOAT_VECTOR | 图像向量(512维) |
| kb_id | VARCHAR(64) | 知识库ID |

#### fusion_vectors 融合向量集合
| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(64) | 记录ID |
| chunk_id | VARCHAR(64) | 分块ID |
| fusion_vector | FLOAT_VECTOR | 融合向量(1024维) |
| kb_id | VARCHAR(64) | 知识库ID |

---

## 核心关联关系

```sql
-- 用户与知识库关联
users.user_id → knowledge_bases.owner_id

-- 知识库与文档关联  
knowledge_bases.kb_id → documents.kb_id

-- 文档与内容分块关联
documents.doc_id → content_chunks.doc_id

-- 内容分块与向量关联
content_chunks.chunk_id → vector_metadata.chunk_id
content_chunks.chunk_id → Milvus集合.chunk_id
```

## 核心索引策略

### 关系数据库索引
- `users(user_id)`
- `knowledge_bases(kb_id, owner_id)`  
- `documents(doc_id, kb_id)`
- `content_chunks(chunk_id, doc_id, kb_id)`
- `vector_metadata(chunk_id, collection_name)`

### Milvus索引
- 向量字段: IVF_FLAT/COSINE相似度
- 标量字段: `chunk_id`, `kb_id` 倒排索引

## 核心数据流

1. **用户上传文档** → `documents`表记录
2. **文档解析分块** → `content_chunks`表存储
3. **向量化处理** → `vector_metadata` + Milvus集合
4. **检索查询** → 多模态向量相似度搜索
5. **返回结果** → 关联内容分块详细信息

这种简化架构保持了：
- **清晰的层级关系**
- **高效的多模态检索**  
- **灵活的扩展能力**
- **完整的数据追溯**