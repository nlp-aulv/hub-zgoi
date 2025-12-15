from agents import Agent
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from fastmcp import Client
import asyncio
import pdfplumber
from typing import List,Dict,Tuple
from sklearn.metrics.pairwise import cosine_similarity
from agents import Agent, Runner, OpenAIChatCompletionsModel, ModelSettings
from agents.extensions.memory import AdvancedSQLiteSession
from agents.mcp import MCPServerSse, ToolFilterStatic
from openai import AsyncOpenAI

def semantic_search(query:str, docs:Dict[str,str],top_k=5) ->  List[Tuple[str, str, float]]:
    model = SentenceTransformer('../models/Qwen3-Embedding-0.6B')
    # 编码
    query_encode = model.encode([query])
    docs_encode = model.encode(list(docs.values()))

    # 计算相似度
    similarities = cosine_similarity(
        query_encode,
        docs_encode
    )[0]

    # 创建包含键、值和相似度的元组列表
    results = []
    for (key, value), similarity in zip(docs.items(), similarities):
        results.append((key, value, similarity))

    # 按相似度降序排序
    results.sort(key=lambda x: x[2], reverse=True)

    # 返回前k个结果
    return results[:top_k]


def rerank(query:str, docs:List[Tuple[str, str, float]],top_k=3) -> List[Tuple[str, str, float]]:
    # 加载rerank模型和分词器
    rerank_path = "../models/Qwen3-Reranker-0.6B/models/Qwen3-Reranker-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(rerank_path)
    model = AutoModelForCausalLM.from_pretrained(rerank_path).eval()
    max_length = 8192
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")

    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)

    task = 'Given a web search query, retrieve relevant passages that answer the query'
    def format_instruction(instruction, query, doc):
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
        output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,
                                                                                         query=query, doc=doc)
        return output

    def process_inputs(pairs):
        inputs = tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
        inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(model.device)
        return inputs

    @torch.no_grad()
    def compute_logits(inputs, **kwargs):
        batch_scores = model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    # Tokenize the input texts
    pairs = [format_instruction(task, query, doc[1]) for doc in docs]
    inputs = process_inputs(pairs)
    scores = compute_logits(inputs)

    results = []
    for t, score in zip(docs, scores):
        results.append((t[0], t[1], score))

    # 按相似度降序排序
    results.sort(key=lambda x: x[2], reverse=True)

    # 返回前k个结果
    return results[:top_k]

def read_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

async def ask_with_agent(query:str):
    # 数据准备
    mcp_server = MCPServerSse(
        name="SSE Python Server",
        params={"url": "http://127.0.0.1:8900/sse"},
        cache_tools_list=False,
        client_session_timeout_seconds=20,
    )
    async with mcp_server:
        mcp_infos = await mcp_server.list_tools()
        tools = rerank(query,semantic_search(query, {tool.name:tool.description for tool in mcp_infos }))
        filter_tools = [tool[0] for tool in tools]
        print(filter_tools)
        mcp_tools_filter: ToolFilterStatic = ToolFilterStatic(allowed_tool_names=filter_tools)
        mcp_server.tool_filter = mcp_tools_filter
        # 智能体
        api_key = "sk-"
        base_url = "https://api.deepseek.com"
        model_name = "deepseek-chat"
        instructions = f"""
                你是一个专业的数学模型人员，可以根据用户的问题和提供的资料回答问题，也可以调用工具计算数学模型的结果。
            """
        open_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        agent = Agent(
            name="Assistant",
            instructions=instructions,
            model=OpenAIChatCompletionsModel(
                model=model_name,
                openai_client=open_client,
            ),
            mcp_servers=[mcp_server],
            # tool_use_behavior='stop_on_first_tool',
            model_settings=ModelSettings(parallel_tool_calls=False)
        )

        # 运行
        prompt = f"""
            用户的问题：{query}
            
            资料如下：
            """
        for idx, t in enumerate(tools):
            prompt += f"{idx}: {t[1]}\n"


        result = await Runner.run(agent, input=prompt)
        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(ask_with_agent("我现在有两个变量x和y，x=100,y=50,我想知道这两个变量可能的作用程度值多少"))
