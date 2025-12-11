import os
import re
import glob
import traceback
from openai import OpenAI
from sentence_transformers import SentenceTransformer

print("rag问答系统启动...")


class RAGMinerU:
    def __init__(self, data_dir=''):
        self.data_dir = data_dir
        
        # 获取data_dir目录下第一个.md文件名
        try:
            # self.mineru_data(input_path, out_path)
            self.md_files_dir = glob.glob(os.path.join(self.data_dir, '*.md'))
            self.md_file_dir = self.md_files_dir[0] if self.md_files_dir else None
            if self.md_file_dir:
                print(f"找到MinerU解析的第一个.md文件: {self.md_file_dir}")
            else:
                print("未找到MinerU解析的.md文件")
        except Exception as e:
            print(f"获取data_dir目录下第一个.md文件名报错: {e}")
            traceback.print_exc()
            raise

        # 加载模型
        try:
            # self.tokenizer = AutoTokenizer.from_pretrained("E:/AI_study/models/BAAI/bge-small-zh-v1.5")
            self.model = SentenceTransformer("E:/AI_study/models/BAAI/bge-small-zh-v1.5")  # bge-small-zh-v1.5可能对中英文互相匹配的效果不太好，可以换一个模型
            print("模型加载完成")
        except Exception as e:
            print(f"加载模型报错: {e}")
            traceback.print_exc()
            raise
        
        # 加载文档
        try:
            with open(self.md_file_dir, 'r', encoding='utf-8') as file:
                self.document = file.read()
            print(f"文档加载完成, 共{len(self.document)}个字符")
        except Exception as e:
            print(f"加载文档报错: {e}")
            traceback.print_exc()
            raise

        # 文档分块
        try:
            self.document_chunks = self.split_document()
            print(f"文档分块完成, 共{len(self.document_chunks)}个块")
        except Exception as e:
            print(f"文档分块报错: {e}")
            traceback.print_exc()
            raise

        # 块向量化
        try:
            self.chunks_embedding = self.model.encode(self.document_chunks, normalize_embeddings=True)
            print(f"块向量化完成, 形状: {self.chunks_embedding.shape}")
        except Exception as e:
            print(f"块向量化报错: {e}")
            traceback.print_exc()
            raise



    def mineru_data(self, input_path, out_path, mineru_model_source_='local'):
        # 使用mineru解析文档
        # input_path = r"E:\AI_study\Week15\模型论文\2509-MinerU2.5.pdf"
        # output_path = r"E:\AI_study\Week15\Week15_xx\data"
        # os.environ["MINERU_MODEL_SOURCE"] = "modelscope"
        os.system(f"mineru -p {input_path} -o {out_path}  --source {mineru_model_source_}")

    def split_document(self, chunk_size=500, overlap=100):
        """分割文档为小块"""
        # 按段落分割
        paragraphs = re.split(r'\n\n+', self.document)
        
        chunks = []
        current_chunk=""
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            if len(current_chunk) + len(para) + 1 <= chunk_size:
                current_chunk += para + "\n"
                continue
            else:
                chunks.append(current_chunk.strip())
                current_chunk = current_chunk[-overlap:] + para + "\n" # 保留overlap个字符作为重叠部分
                while len(current_chunk) > chunk_size: # 确保当前块不超过chunk_size
                    chunks.append(current_chunk[:chunk_size].strip())
                    current_chunk = current_chunk[chunk_size-overlap:] # 移除前chunk_size-overlap个字符
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def query(self, query, top_k=3):
        """检索与查询相关的文档块"""
        # 对查询进行向量化
        query_embedding = self.model.encode(query, normalize_embeddings=True)
        
        # 计算查询与文档块的相似度
        similarities = query_embedding @ self.chunks_embedding.T
        
        # 获取top_k个最相关的文档块
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_chunks = [self.document_chunks[i] for i in top_indices]
        
        return top_chunks

    def answer(self, query, top_k=3):
        """根据查询生成答案"""
        # 检索相关文档块
        relevant_chunks = self.query(query, top_k)
        
        # 合并文档块
        context = "\n\n".join(relevant_chunks)
        
        # 生成答案
        prompt = f"根据以下内容回答问题:\n{context}\n\n问题: {query}"
        print(prompt)
        client = OpenAI(
            api_key="",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        answer_text=''
        print("-------------------------")
        for resp in client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        ):
            if resp.choices and resp.choices[0].delta.content:
                answer_text += resp.choices[0].delta.content
                print(resp.choices[0].delta.content, end="", flush=True)        
        return answer_text

    def chat(self):
        """交互式问答"""
        print("\n=== MinerU2.5 RAG问答系统 ===")
        print("输入'quit'退出系统")
        
        while True:
            query = input("\n请输入您的问题: ")
            if query.lower() == 'quit':
                break
            
            answer_text = self.answer(query)
            print("\n -------------------------")
            print("\n回答:")
            print(answer_text)

if __name__ == "__main__":
    # 初始化RAG系统
    rag_system = RAGMinerU("e:\\AI_study\\Week15\\Week15_xx\\data\\2509-MinerU2.5\\auto")
    
    # 开始交互式问答
    rag_system.chat()

