import requests

url = "http://localhost:11434/api/generate"

payload = {
    "model": "qwen3:0.6b",
    "prompt": "解释下rag主要的流程",
    "stream": False
}
response = requests.post(url, json=payload)
print(response.json()['response'])
