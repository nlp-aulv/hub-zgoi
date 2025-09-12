

# Please install OpenAI SDK first: `pip3 install openai` sk-be9fcf311*****7f31f71fc6997e

from openai import OpenAI

client = OpenAI(api_key="sk-be9fcf311*****1fc6997e", base_url="https://api.deepseek.com")

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ],
    stream=False
)

print(response.choices[0].message.content)
