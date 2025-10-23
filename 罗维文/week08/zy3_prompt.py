import openai
import json
from pydantic import BaseModel, Field
from typing import Union, List, Dict, Optional

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
      "待选实体": ["实体名词1","实体名词2",...],
    }}
}}
```
"""


def model_for_prompt(request_text: Union[str, List[str]]) -> List[Dict[str, Union[str, Dict[str, List[str]]]]]:
    texts = [request_text] if isinstance(request_text, str) else request_text
    results = []

    for text in texts:
        if not text or not text.strip():
            results.append({"text_error": "空文本", "domain": "", "intent": "", "slots": {}})
            continue

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text.strip()}
                ],
            )
            response = completion.choices[0].message.content
            print(response)
            # 解析 LLM 输出
            result = parse_llm_output(response)
            results.append(result)

        except Exception as e:
            results.append({
                "text_error": f"模型调用失败: {str(e)}",
                "domain": "",
                "intent": "",
                "slots": {}
            })

    return results


def parse_llm_output(output: str) -> Dict[str, Union[str, Dict[str, List[str]]]]:
    """
    安全解析 LLM 输出的 JSON，支持 ```json 包裹的情况
    """
    try:
        # 提取代码块中的 JSON
        if '```json' in output:
            json_str = output.split('```json')[1].split('```')[0].strip()
        elif '```' in output:
            json_str = output.split('```')[1].strip()
        else:
            json_str = output.strip()

        data = json.loads(json_str)

        # 标准化字段
        domain = data.get("domain", "").strip() or ""
        intent = data.get("intent", "").strip() or ""
        raw_slots = data.get("slots", {})
        if not isinstance(raw_slots, dict):
            raw_slots = {}

        slots = {
            k: [str(x).strip() for x in (v if isinstance(v, list) else [v] if v is not None else [])]
            for k, v in raw_slots.items()
            if isinstance(k, str)
        }
        return {"domain": domain, "intent": intent, "slots": slots}

    except Exception as e:
        raise ValueError(f"解析失败: {e}, 原始输出:\n{output}")

