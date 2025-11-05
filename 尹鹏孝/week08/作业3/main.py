from fastapi import FastAPI, Request, HTTPException
from typing import List, Optional, Dict, Any
from llm_tools import llm_tools
from prompt_domain_intent_slots import prompt_domain_intent_slots

app = FastAPI(
    title="领域 、意图、实体识别",
    description="基于大模型的领域 、意图、实体识别的api",
    version="1.0.0"
)


# 使用 Request 对象手动获取参数
@app.get("/llm/indent/tools/", response_model=Dict[str, Any])
async def get_llm_indent(request: Request):
    # 手动获取查询参数
    query = request.query_params.get("query")

    print("在get_llm_indent中，query的类型是:", type(query))
    print('入口的查询:', query)

    try:
        """获取类型识别"""
        if query is None or query.strip() == "":
            return {
                "code": 400,
                "message": "请输入查询条件"
            }

        result = prompt_domain_intent_slots(query)
        print('处理结果:', result)

        # 确保结果是字典
        if not isinstance(result, dict):
            result = {"response": str(result)}

        result.update({"code": 200})  # 添加键值对
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# prompts
@app.get("/llm/indent/prompts/", response_model=Dict[str, Any])
async def get_llm_indent_prompts(request: Request):
    # 手动获取查询参数
    query = request.query_params.get("query")
    print(222)
    try:
        if query is None or query.strip() == "":
            return {
                "code": 400,
                "message": "请输入查询条件"
            }

        """获取类型识别"""
        result = llm_tools(query)
        print(result)
        # 确保结果是字典
        if not isinstance(result, dict):
            result = {"response": str(result)}

        result.update({"code": 200})
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)