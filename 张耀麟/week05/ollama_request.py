import requests
import json


def call_ollama_api(model, user_input):
    data = {
        "model": model,
        "prompt": user_input,
        "stream": False  # 是否以流式返回（可设为 True/False）
    }

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data
        )
        response.raise_for_status()  # 如果请求返回了不成功的状态码，抛出异常

        with open(r"./ollama_response.json", "w+", ) as file:
            json.dump(response.json(), file, )

        return response.json()  # 解析JSON响应
    except requests.exceptions.RequestException as e:
        print(f"请求过程中发生错误：{e}")
        return None


if __name__ == "__main__":
    model = "qwen3:0.6B"
    call_ollama_api(model, "介绍一下git")

    with open(r"./ollama_response.json", "r", ) as fp:
        json = json.load(fp)
        print(json["response"])
