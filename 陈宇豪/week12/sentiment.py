import requests
from openai import OpenAI

TOKEN = "sk-2768f88e42fb47c0ba9d9498b45c9c78"

from fastmcp import FastMCP

mcp = FastMCP(
    name="Sentiment analysis service",
    instructions="""This server contains some api of Sentiment analysis.""",
)


@mcp.tool
def sentiment_analysis(text: str) -> str:
    """
    使用Qwen大模型进行情感分析
    
    参数:
        text: 需要进行情感分析的文本
        
    返回:
        情感分析结果
    """
    try:
        # 使用DashScope的Qwen模型进行情感分析
        client = OpenAI(
            api_key=TOKEN,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )

        response = client.chat.completions.create(
            model="qwen-max",
            messages=[
                {
                    "role": "system",
                    "content": "你是一个专业的情感分析助手。请分析给定文本的情感倾向，判断是正面、负面还是中性情感，并给出简要解释。"
                },
                {
                    "role": "user",
                    "content": f"请分析以下文本的情感倾向：{text}"
                }
            ],
            temperature=0.7,
            max_tokens=500
        )

        return response.choices[0].message.content
    except Exception as e:
        return f"情感分析过程中出现错误: {str(e)}"


if __name__ == '__main__':
    # print(sentiment_analysis("我非常开心"))
    pass
