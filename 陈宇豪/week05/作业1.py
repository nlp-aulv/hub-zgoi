import traceback

from openai import OpenAI

from cache import LRUMemoryKVStore
from common import *

client_deepseek = OpenAI(api_key="sk-3fdaf066********e10bb241d003b", base_url="https://api.deepseek.com")
chatMap = LRUMemoryKVStore()
MAX_LENGTH = 10


def deep_seek_chat(req: LlmRequest) -> LlmResponse:
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
        response_deepseek = client_deepseek.chat.completions.create(
            model="deepseek-chat",
            messages=history_chat,
            stream=False
        )
        response.response_text = response_deepseek.choices[0].message.content
        history_chat.append({"role": "system", "content": response.response_text})
        chatMap.set(user, history_chat)
    except Exception as err:
        # error 日志
        response.response_text = "抱歉，请稍后重试"
        response.error_msg = traceback.format_exc()
    return response
