import openai

client = openai.OpenAI(
    api_key='',
    base_url='http://localhost:11434/v1'
)

for resp in client.chat.completions.create(
    model='qwen3:0.6b',
    messages=[
        {'role': 'user', 'content': '你是谁'},
        {'role': 'user', 'content': '你能背乘法表吗'}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end='', flush=True)

