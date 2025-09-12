import openai

client = openai.OpenAI(
    api_key="sk-WkCbMVOViwqUVVdD********071A40213E24c2989Ab",
    base_url="https://openkey.cloud/v1"
)

for resp in client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": "什么是伯劳鸟"}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end="", flush=True)
