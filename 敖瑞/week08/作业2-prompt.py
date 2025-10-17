import openai


def prompt_ner(query):
    client = openai.OpenAI(
        api_key='sk-5070859462a14565a7d30fa7778267b2',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )

    with open('domains.txt', 'r', encoding='utf8') as f:
        domains = ' | '.join([i.replace('\n', '') for i in f.readlines()])

    with open('intents.txt', 'r', encoding='utf8') as f:
        intents = ' | '.join([i.replace('\n', '') for i in f.readlines()])

    with open('slots.txt', 'r', encoding='utf8') as f:
        slots = ' | '.join([i.replace('\n', '') for i in f.readlines()])

    completion = client.chat.completions.create(
        model='qwen-plus',
        messages=[
            {'role': 'user', 'content': """你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类别、实体标签
    - 待选的领域类别：{domains}
    - 待选的意图类别：{intents}
    - 待选的实体标签：{slots}

    最终输出格式填充下面的json，domain 是 领域标签，intent 是 意图标签，slots 是 实体识别结果和标签。
    '''
    {
        "domain": ,
        "intent": ,
        "slots": {
            "待选实体": "实体名词"
        }
    }
    '''
            """},
            {'role': 'user', 'content': query},
        ],
    )

    result = completion.choices[0].message.content

    return result
