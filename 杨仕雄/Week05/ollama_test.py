from openai import OpenAI

client = OpenAI(
    base_url='http://127.0.0.1:11434/v1/',
    api_key='ollama',  # required but ignored
)
chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "你是一个大模型专家，请你解答用户的关于大模型问题",
        },
        {
            "role": "user",
            "content": "大模型中会对模型做微调，请你解释什么情况下要做微调，怎么微调，以及如何判断微调的效果",
        }
    ],

    model='qwen3:0.6b',
    # model='qwen2.5:7b',

    max_tokens=38192,
    temperature=0.7,
    top_p=0.5,
    frequency_penalty=0,
    presence_penalty=2,
)

# print(chat_completion)
print(chat_completion.choices[0].message.content)
