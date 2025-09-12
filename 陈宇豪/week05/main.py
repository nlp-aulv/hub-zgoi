from openai import OpenAI
from cache import LRUMemoryKVStore
from fastapi import FastAPI
import uvicorn
import traceback
from common import *
from 作业1 import deep_seek_chat
from 作业2 import local_qwen_chat

app = FastAPI(
    title="简单的大模型调用",
    description="基于大模型api调用",
    version="1.0.0"
)


@app.post("/v1/llmChat/deepSeek")
def deep_seek(req: LlmRequest) -> LlmResponse:
    response = deep_seek_chat(req)
    return response


@app.post("/v1/localChat/qwen")
def local_qwen(req: LlmRequest) -> LlmResponse:
    response = local_qwen_chat(req)
    return response


if __name__ == '__main__':
    # 运行fastapi程序
    uvicorn.run(app="main:app", host="127.0.0.1", port=8000, reload=True)
