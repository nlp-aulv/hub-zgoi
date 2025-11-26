字段名	数据类型	是否主键	是否向量	描述
id	VARCHAR	是	否	全局唯一ID，格式如 doc001_page002
document_id	VARCHAR	否	否	所属文档的ID，如 doc001
page_number	Int64	否	否	页码或顺序号
modality_type	VARCHAR	否	否	用于区分数据类型：text, image, table
text_content	VARCHAR	否	否	纯文本内容（用于文本模态，也作为图像/表格的OCR或描述备用）
text_vector	FLOAT_VECTOR	否	是	文本嵌入向量（dim=文本模型维度，如1024）
image_vector	FLOAT_VECTOR	否	是	图像嵌入向量（dim=视觉模型维度，如768）
table_json	JSON	否	否	表格的结构化数据，存储为JSON字符串
table_vector	FLOAT_VECTOR	否	是	表格内容的向量表示（dim=文本模型维度）
raw_data_path	VARCHAR	否	否	原始数据路径（如图片文件路径、表格截图路径等）
