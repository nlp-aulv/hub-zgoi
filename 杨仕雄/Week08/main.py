# 第三方库
import openai
from fastapi import FastAPI
from pydantic import BaseModel, Field

# 自己写的模块
from prompt import prompt_test
from tools import tool_test

app = FastAPI()

class Req(BaseModel):
    text: str

@app.post("/v1/text-cls/prompt")
def prompt_test(req:Req):

    response = prompt_test(req.text)
    return response

@app.post("/v1/text-cls/tools")
def tools(req:Req):

    response = tool_test(req.text)
    return response