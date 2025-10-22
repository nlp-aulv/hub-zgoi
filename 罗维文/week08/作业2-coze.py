import requests
import json

# url = "https://api.coze.cn/open_api/v1/chat"
# url = 'https://api.coze.cn/v1/conversation/create'
url = "https://api.coze.cn/v3/chat"
headers = {
    "Authorization": "Bearer pat_HJDiZRFELMUVITtMWqqAj3IUIYBzmK4EO9ZRujEH72tg1wWY4Eux2z1uv3GDnQyT",
    "Content-Type": "application/json",
}
data = {
    "bot_id": "7561547780750278675",  # 替换为你的 bot_id
    "workflow_id": "7561561852367585280",
    "user_id": "newid1234",
    "stream": True,

    # "auto_save_history": True,
    "additional_messages":[
        {
            "role": "user",
            "content": "把李志强的号码发给贾洪鉴",
            "content_type": "text"
        }
    ],
    # "query": "查询北京飞桂林的飞机是否已经起飞了"
}

response = requests.post(url, headers=headers, json=data)
# print(response.text)
# print(response.status_code)

if response.status_code == 200:
    lines_iter = response.iter_lines(decode_unicode=True)
    for line in lines_iter:
        if line.strip():
            if "completed" in line:
                line = next(lines_iter)
                if line.startswith("data:"):
                    data_str = line[5:].strip()
                    try:
                        data_json = json.loads(data_str)
                        if data_json.get("type") == "answer":
                            content = data_json["content"]
                            content = json.loads(content)
                            for k, v in content["slots"].items():
                                after_v = [x.encode('latin1').decode('utf-8') for x in v]
                                content["slots"][k] = after_v
                            print(content, end="", flush=True)
                            break
                    except Exception as e:
                        print(f"\n解析行失败: {e}")
            if "event: conversation.chat.failed" in line:
                print(f"\n响应失败: {e}")
else:
    print("调用失败:", response.status_code, response.text)