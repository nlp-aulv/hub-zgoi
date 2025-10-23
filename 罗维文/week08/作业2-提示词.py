import openai
import json

# https://bailian.console.aliyun.com/?tab=api#/api/?type=model&url=2712576
client = openai.OpenAI(
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-78cc4e9ac8f44efdb207b7232e1ae6d8",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
model="qwen-plus"
# 领域类别
with open('E:/AI学习/02-week08-joint-bert-training-only/domains.txt', 'r', encoding='utf-8') as fp:
    domains = fp.read().split('\n')
    domains = ' / '.join(domains)

# 意图类别
with open('E:/AI学习/02-week08-joint-bert-training-only/intents.txt', 'r', encoding='utf-8') as fp:
    intents = fp.read().split('\n')
    intents = ' / '.join(intents)

# 实体标签
with open('E:/AI学习/02-week08-joint-bert-training-only/slots.txt', 'r', encoding='utf-8') as fp:
    slots = fp.read().split('\n')
    slots = ' / '.join(slots)

system_prompt=f"""你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
- 待选的领域类别：{domains}
- 待选的意图类别：{intents}
- 待选的实体标签：{slots}

最终输出格式填充下面的json， domain 是 领域标签， intent 是 意图标签，slots 是实体识别结果和标签。

```json
{{
    "domain": ,
    "intent": ,
    "slots": {{
      "待选实体": "实体名词",
    }}
}}
```
"""

with open('E:/AI学习/02-week07-joint-bert-training-only/data/test.json', 'r', encoding='utf-8') as fp:
    pred_data = eval(fp.read())
    for i, p_data in enumerate(pred_data):
        text = p_data['text']
        print('=================================')
        print(text)
        messages = [{"role": "system", "content": system_prompt}
                    ,{"role": "user", "content": text}]
        completion = client.chat.completions.create(
            model=model,
            messages=messages
        )
        result = completion.choices[0].message.content
        print('=================================')
        print(result)
        if i == 10:
            break