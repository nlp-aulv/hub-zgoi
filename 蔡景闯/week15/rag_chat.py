from sentence_transformers import SentenceTransformer
import torch
from sklearn.metrics.pairwise import cosine_similarity
from typing import List,Dict,Tuple
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
import asyncio
import json

from sympy.physics.units import current


def semantic_search(query:str, docs:List[str],top_k=5) ->  List[str]:
    model = SentenceTransformer('../models/Qwen3-Embedding-0.6B')
    # 编码
    query_encode = model.encode([query])
    docs_encode = model.encode(list(docs))

    # 计算相似度
    similarities = cosine_similarity(
        query_encode,
        docs_encode
    )[0]

    # 获取最相似的top_k个文档
    top_k_indices = similarities.argsort()[-top_k:][::-1] # 从大到小排序
    return [docs[i] for i in top_k_indices]

def get_docs() -> Tuple[List[str],List[str]]:
    with open('./1_content_list.json', 'r', encoding='utf-8') as f:
        content_list = json.load(f)

        text_list = []
        formula_list = []

        for item in content_list:
            # 处理公式
            if item.get('type') == 'equation' and item.get('text_format') == 'latex':
                formula_list.append(item['text'])
            # 处理文本
            elif item.get('type') == 'text':
                text = item['text']
                if len(text) <= 100:
                    text_list.append(text)
                else:
                    # 使用句号分割，拼接后不超过100个字
                    current = ""
                    sentences = text.split('。')
                    for sentence in sentences:
                        if len(current + sentence) <= 100:
                            current += sentence + '。'
                        else:
                            text_list.append(current)
                            current = sentence + '。'

    return  text_list, formula_list

async def chat(query):
    #rag
    docs, formulas = get_docs()
    rag_result = semantic_search(query, docs)
    print(rag_result)

    #agent
    api_key = "sk-"
    base_url = "https://api.deepseek.com"
    model_name = "deepseek-chat"
    instructions = f"""
                        你是一个专业的数学模型人员，可以根据用户的问题和提供的资料回答问题。
                    """

    open_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    agent = Agent(
        name="Assistant",
        instructions=instructions,
        model=OpenAIChatCompletionsModel(
            model=model_name,
            openai_client=open_client,
        ),
        # tool_use_behavior='stop_on_first_tool',
        model_settings=ModelSettings(parallel_tool_calls=False)
    )

    # 运行
    prompt = f"""
用户的问题：{query}

资料如下：
            """
    for r in rag_result:
        prompt += f"{r}\n"

    for formula in formulas:
        prompt += f"\n涉及的公式：\n{formula}\n"

    print(prompt)

    result = await Runner.run(agent, input=prompt)
    print(result.final_output)


if __name__ == '__main__':
    asyncio.run(chat("金融服务中，有没有办法对价格进行预测"))