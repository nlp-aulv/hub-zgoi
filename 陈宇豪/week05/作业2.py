import traceback

from openai import OpenAI

from cache import LRUMemoryKVStore
from common import LlmRequest, LlmResponse

client_qwen = OpenAI(api_key="qwen", base_url="http://localhost:11434/v1")
chatMap = LRUMemoryKVStore()
MAX_LENGTH = 10


def local_qwen_chat(req: LlmRequest) -> LlmResponse:
    response = LlmResponse(
        request_user=req.request_user,
        error_msg="",
        response_text="",
    )
    # info 日志
    try:
        user = req.request_user
        history_chat = chatMap.get(user)
        if history_chat is None:
            history_chat = [
                {"role": "user", "content": req.request_text},
            ]
        else:
            history_chat.append({"role": "user", "content": req.request_text})
        if len(history_chat) > MAX_LENGTH:
            history_chat = history_chat[-MAX_LENGTH:]
        response_qwen = client_qwen.chat.completions.create(
            model="qwen3:0.6b",
            messages=history_chat,
            stream=False,
            temperature=0.7,  # 控制生成多样性
            max_tokens=512  # 最大生成 token 数
        )
        response.response_text = response_qwen.choices[0].message.content
        history_chat.append({"role": "system", "content": response.response_text})
        chatMap.set(user, history_chat)
    except Exception as err:
        # error 日志
        response.response_text = "抱歉，请稍后重试"
        response.error_msg = traceback.format_exc()
    return response
