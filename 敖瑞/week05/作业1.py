import openai

client = openai.OpenAI(
    api_key='sk-46c6119ce42****57fdac83a',
    base_url='https://api.deepseek.com/v1'
)

for resp in client.chat.completions.create(
    model='deepseek-chat',
    messages=[
        {'role': 'user', 'content': '背一下99乘法表'}
    ],
    stream=True
):
    if resp.choices and resp.choices[0].delta.content:
        print(resp.choices[0].delta.content, end='', flush=True)

