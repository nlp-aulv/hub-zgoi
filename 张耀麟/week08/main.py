from fastapi import FastAPI
from prompt_model import prompt
from tools_model import tools
import uvicorn

app = FastAPI()

# 路由分发include_router
app.include_router(prompt, prefix="/prompt", tags=["提示词识别接口"])
app.include_router(tools, prefix="/tools", tags=["tools识别接口"])


if __name__ == '__main__':
    uvicorn.run(app, host="127.0.0.1", port=8000)
