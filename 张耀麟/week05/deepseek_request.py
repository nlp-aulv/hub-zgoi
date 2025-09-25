import requests
import json


def call_deepseek_api(api_key, user_input):
    url = "https://api.deepseek.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "deepseek-chat",  # 指定模型
        "messages": [
            {"role": "system", "content": "你是一个有帮助的助手。"},  # 可选的系统消息，设定助手角色
            {"role": "user", "content": user_input}
        ],
        "temperature": 0.7,  # 控制生成随机性
        "max_tokens": 100,  # 控制生成内容的最大长度
        "stream": False  # 非流式
    }

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 如果请求返回了不成功的状态码，抛出异常
        with open(r"./deepseek_response.json", "w+", ) as file:
            json.dump(response.json(), file, )
        # print(response.json())
        return response.json()  # 解析JSON响应
    except requests.exceptions.RequestException as e:
        print(f"请求过程中发生错误：{e}")
        return None


if __name__ == "__main__":
    api_key = "sk-0d5c48a655774117bf8e4dfea9eb9f7f"

    # call_deepseek_api(api_key, "介绍一下git")

    with open(r"./deepseek_response.json", "r", ) as fp:
        json = json.load(fp)
        for choice in json["choices"]:
            print(choice["message"]["content"])
