# -*- coding: utf-8 -*-
import openai
import os
from dotenv import load_dotenv
import re
# 加载环境变量（密钥放在.env文件中）
load_dotenv()

def replace_newlines_simple(file_path):
    """读取文件并替换换行符"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        # 替换所有换行符为 ' / '
        content = content.replace('\n', ' / ')
        return content
    except FileNotFoundError:
        print(f"文件 {file_path} 未找到")
        return ""
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return ""


# 加载数据
domain = replace_newlines_simple('domains.txt')
intent = replace_newlines_simple('intents.txt')
slots = replace_newlines_simple('slots.txt')

# 初始化客户端（
client = openai.OpenAI(
    api_key=os.getenv('OPENAI_API_KEY', "sk-411bf89559914810893fd40f59a24515"),
    base_url= os.getenv('OPENAI_BASE_URL', "https://dashscope.aliyuncs.com/compatible-mode/v1"),
)

# 构建消息
system_message = {
    "role": "system",
    "content": f"""你是一个专业信息抽取专家，请对下面的文本抽取他的领域类别、意图类型、实体标签
        - 待选的领域类别：{domain}
        - 待选的意图类别：{intent} 
        - 待选的实体标签：{slots}

        最终输出格式填充下面的json：
        ```json
        {{
            "domain": "领域名称",
            "intent": "意图名称",
            "slots": {{
                "实体类型": "实体值"
            }}
        }}
        ```

        请确保返回有效的JSON格式。"""
}

user_message = {
    "role": "user",
    "content": "帮我查询2025年10月15日西安到北京的高铁"
}
def prompt_domain_intent_slots(query='帮我查询2025年10月15日西安到北京的高铁'):
    user_message['content'] = query
    try:
        completion = client.chat.completions.create(
            model="qwen-plus",
            messages=[system_message, user_message],
        )

        # 正确获取回复内容
        if completion.choices and len(completion.choices) > 0:
            result_content = completion.choices[0].message.content
            print("模型回复:")
            print(result_content)

            # 尝试提取JSON部分
            json_match = re.search(r'```json\n(.*?)\n```', result_content, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                print("\n提取的JSON:")
                print(json_str)
            else:
                print("\n直接输出:")
                print(result_content)

        else:
            print("未收到有效回复")

    except Exception as e:
        result_content = e
        print(f"API调用失败: {result_content}")

    return user_message

prompt_domain_intent_slots()