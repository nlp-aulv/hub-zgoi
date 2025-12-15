#根据思路二弄的但感觉问题很大
import torch
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFacePipeline
# 替代 UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
import time

# 首先解析pdf和md文件弄成csv文件
def load_document_loaders():
    try:
        from langchain_community.document_loaders import PyPDFLoader
        return PyPDFLoader
    except ImportError as e:
        print(f"导入PyPDFLoader失败: {e}")
        # 提供备用方案
        class DummyLoader:
            def __init__(self, file_path):
                self.file_path = file_path
            def load(self):
                return []
        return DummyLoader

# 你的文件夹路径
folder_path = "documents"

records = []

# 获取PDF加载器
PyPDFLoader = load_document_loaders()

# 遍历文件夹里的所有文件
for filename in tqdm(os.listdir(folder_path)):
    file_path = os.path.join(folder_path, filename)

    if filename.lower().endswith(".pdf"):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            # 拼接所有页
            full_text = "\n".join(doc.page_content for doc in documents)
            records.append({"filename": filename, "content": full_text})
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            records.append({"filename": filename, "content": ""})

    elif filename.lower().endswith(".md"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                full_text = f.read()
            records.append({"filename": filename, "content": full_text})
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            records.append({"filename": filename, "content": ""})

# 保存成 CSV，每个文件一行
try:
    df = pd.DataFrame(records)
    df.to_csv("match.csv", index=False, encoding="utf-8-sig")
    print(f"已处理 {len(records)} 个文件，保存到 match.csv")
except Exception as e:
    print(f"保存CSV文件时出错: {e}")


#然后弄大模型
os.environ["http_proxy"] = "http://127.0.0.1:7890"
os.environ["https_proxy"] = "http://127.0.0.1:7890"
os.system('export HF_ENDPOINT=https://hf-mirror.com')
import csv

def append_to_csv(file_path, col1, col2):
    # 判断文件是否存在
    file_exists = os.path.isfile(file_path)

    # 以追加模式打开
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # 如果文件不存在，先写表头
        if not file_exists:
            writer.writerow(['Column1', 'Column2'])
        # 写入一行数据
        writer.writerow([col1, col2])

# ========== 加载本地 Qwen 模型 ==========
model_name = "qwen3"
t1 = time.time()
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

# HuggingFace pipeline 封装
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7
)
llm = HuggingFacePipeline(pipeline=pipe)
t2 = time.time()
print(t2 - t1,"t2 - t1")
# ========== Part 1: PDF 知识库问答 ==========
# 加载 PDF

folder = "./matched.csv"

import pandas as pd

df = pd.read_csv(folder)
first_col_list = df.iloc[:, 0].tolist()   # 取第一列并转成list
second_col_list = df.iloc[:, 1].tolist()   # 取第二列并转成list
print(first_col_list)
match = []
for i in range(len(first_col_list)):
    match.append([first_col_list[i],second_col_list[i]])
print(match[0])
content_list = []
from tqdm import tqdm
for f in tqdm(match):
    prompt = f"请仔细阅读以下数学知识，并用一个数字回答我的数学问题，背景知识，只需要回答一个数字，可以是小数或多位数，如果无法计算但可以估计结果，就回复我：“估计是（估计结果）”，例如你可以根据常识估算出结果在10000左右，就回答我“估计是（10000）“，不需要任何多余的解释或计算过程，我的问题是：{f[0]}，我的背景知识是{f[1]}，再次强调只要回答我不知道或者一个数或者估计结果作为回答，请注意其中“万元”等词语，在回答中要给出对应的阿拉伯数字"

    messages = [
        {"role": "user", "content": prompt}
    ]
    # 适配 Qwen 的对话模板
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    ##print("General Chat:", content)
    content_list.append(content)

df = pd.DataFrame(content_list, columns=["Column1"])
df.to_csv("content_list2.csv", index=False, encoding="utf-8-sig")
