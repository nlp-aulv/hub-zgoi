from fastmcp import FastMCP
from typing import Annotated
from openai import OpenAI

mcp = FastMCP(
    name='Sentiment-MCP-Server',
    instructions="""This server contains some api of sentiment analysis."""
)

client = OpenAI(
    api_key='sk-5070859462a14565a7d30fa7778267b2',
    base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
)


@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """
    Classifies the sentiment of a given text.
    """
    # print('调用工具：sentiment_classification')
    prompt = """分析以下文本的情感倾向，只需返回单个词：'positive'、'negative'或'neutral'。
    """
    try:
        response = client.chat.completions.create(
            model='qwen-max',
            messages=[
                {
                    'role': 'system',
                    'content': prompt
                },
                {
                    'role': 'assistant',
                    'content': text
                }
            ],
            # 开启流式调用
            stream=True,
            # extra_body={
            #     # 开启思考模式
            #     'enable_thinking': True
            # }
        )

        answer_content = ""

        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta

                if hasattr(delta, 'content') and delta.content is not None:
                    answer_content += delta.content

        # result = response.choices[0].message.content.strip().lower()
        if answer_content in ['positive', 'negative', 'neutral']:
            print(answer_content)
            return answer_content
        else:
            print('警告：模型返回非预期值，模型返回内容为：', answer_content)
            return 'neutral'
    except Exception as e:
        print('调用情感分析模型出错，错误信息：', e)
        return 'neutral'
