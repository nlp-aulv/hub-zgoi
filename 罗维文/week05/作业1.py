from openai import OpenAI
import json
client = OpenAI(api_key="<DeepSeek API Key>", base_url="https://api.deepseek.com")

# Round 1
messages = [{"role": "user", "content": "你好"}]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)

print(response.choices[0].message.content)
print('-------------------------------------------')
print(json.dumps(response.to_dict(), indent=4, ensure_ascii=False))

