
from openai import OpenAI
from fastapi import FastAPI
import uvicorn
app = FastAPI()
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-411bf89559914810893fd40f59a24515",

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)



# http://localhost:8000/emotion方法如下：
@app.get("/emotion")
async def read_user_me(q: str, limit: int = 100):
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": f"请对以下文本进行情感分析，输出情感类别：{q}"}
        ]
    )
    print(completion.model_dump_json())
    return {"result": completion.model_dump_json()}
# 运行应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





# 在mcp_server:tool里面追加如下方法：

# @mcp.tool
# def sentiment_classification(text: Annotated[str, "The text to analyze"]):
#     """Classifies the sentiment of a given text."""
#     try:
#         return requests.get(
#             f"http://localhost:8000/emotion?key={TOKEN}&q={text}").json()[
#             "result"]
#     except:
#         return {}
# 重新运行代码





