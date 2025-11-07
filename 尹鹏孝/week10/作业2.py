# 作业2：使用云端qwen-vl模型，完成带文字截图的图，文本的解析转换为文本。
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
# from qwen_vl_utils import process_vision_info
# from PIL import Image, ImageDraw, ImageFont
# from IPython.display import Markdown, display
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "../model/Qwen2___5-VL-7B-Instruct/", torch_dtype="auto", device_map="auto"
# )
# processor = AutoProcessor.from_pretrained("../model/Qwen2___5-VL-7B-Instruct/")
#
# def inference(image_path, prompt, sys_prompt="You are a helpful assistant.", max_new_tokens=4096, return_input=False):
#     image = Image.open(image_path)
#     image_local_path = "file://" + image_path
#     messages = [
#         {"role": "system", "content": sys_prompt},
#         {"role": "user", "content": [
#                 {"type": "text", "text": prompt},
#                 {"image": image_local_path},
#             ]
#         },
#     ]
#     text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     print("text:", text)
#     # image_inputs, video_inputs = process_vision_info([messages])
#     inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt")
#     inputs = inputs.to('cuda')
#
#     output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, output_ids)]
#     output_text = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
#     if return_input:
#         return output_text[0], inputs
#     else:
#         return output_text[0]
#
# image_path = "./img/2/简历.png"
# prompt = "Read all the text in the image."
# image = Image.open(image_path)
# display(image.resize((800,400)))
# ## Use a local HuggingFace model to inference.
# response = inference(image_path, prompt)
# print(response)


#使用了阿里的在线大模型qwen3-vl-plus
from openai import OpenAI
import os

# 初始化OpenAI客户端
client = OpenAI(
    api_key =  "sk-" ,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

reasoning_content = ""  # 定义完整思考过程
answer_content = ""     # 定义完整回复
is_answering = False   # 判断是否结束思考过程并开始回复
enable_thinking = False
# 创建聊天完成请求
completion = client.chat.completions.create(
    model="qwen3-vl-plus",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        # "url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
                    "url": "https://file.100chui.com/files/upload/image/202107/0432ea66-53c9-4e99-8dac-12eb3dedd9a6.jpg"
                    },
                },
                {"type": "text", "text": "完成带文字截图的图，文本的解析转换为文本"},
            ],
        },
    ],
    stream=True,
    # enable_thinking 参数开启思考过程，thinking_budget 参数设置最大推理过程 Token 数
    extra_body={
        'enable_thinking': False,
        "thinking_budget": 81920},

    # 解除以下注释会在最后一个chunk返回Token使用量
    # stream_options={
    #     "include_usage": True
    # }
)

if enable_thinking:
    print("\n" + "=" * 20 + "思考过程" + "=" * 20 + "\n")

for chunk in completion:
    # 如果chunk.choices为空，则打印usage
    if not chunk.choices:
        print("\nUsage:")
        print(chunk.usage)
    else:
        delta = chunk.choices[0].delta
        # 打印思考过程
        if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
            print(delta.reasoning_content, end='', flush=True)
            reasoning_content += delta.reasoning_content
        else:
            # 开始回复
            if delta.content != "" and is_answering is False:
                print("\n" + "=" * 20 + "完整回复" + "=" * 20 + "\n")
                is_answering = True
            # 打印回复过程
            print(delta.content, end='', flush=True)
            answer_content += delta.content

# print("=" * 20 + "完整思考过程" + "=" * 20 + "\n")
# print(reasoning_content)
# print("=" * 20 + "完整回复" + "=" * 20 + "\n")
# print(answer_content)