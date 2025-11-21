### 第十二周作业
- 作业1. 复现现有职能助手代码，环境要有 fastmcp、streamlit、openai-agent 库。
python mcp_server_main.py
streamlit run streamlit_demo.py

- 作业2. 尝试新定义一个工具，进行文本情感分析，输入文本判断文本的情感类别。最终可以在界面通过agent 在对话中调用这个工具
```python
@mcp.tool
def sentiment_classification(text: Annotated[str, "The text to analyze"]):
    """Classifies the sentiment of a given text."""
    pass
```

- 作业3. 尝试需要在对话中选择工具，增加 tool_filter 的逻辑。
    - 查询新闻的时候，只调用news的工具
    - 调用工具的时候，只调用tools的工具
