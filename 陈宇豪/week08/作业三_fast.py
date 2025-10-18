from fastapi import FastAPI
import uvicorn

from Domain import Response, Request

from 作业二_prompt import chat_model_domain

app = FastAPI(title="大模型领域与实体识别API")


@app.post("/recognize/prompt")
def recognize_domain_and_entities(input_data: Request) -> Response:
    try:
        input_text = input_data.request_text
        response = chat_model_domain(input_text)
        return Response(response_text=response, request_user=input_data.request_user, error_msg="")
    except Exception as e:
        return Response(response_text="", request_user="", error_msg=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
