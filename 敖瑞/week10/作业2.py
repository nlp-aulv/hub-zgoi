import os
from openai import OpenAI
from dashscope import MultiModalConversation


# url链接
def url_image_client(prompt, image_url):

    try:
        client = OpenAI(
            api_key='sk-5070859462a14565a7d30fa7778267b2',
            base_url='https://dashscope.aliyuncs.com/compatible-mode/v1',
        )

        completion = client.chat.completions.create(
            model='qwen-vl-ocr',
            messages=[
                {
                    'role': 'user',
                    'content': [
                        {
                            'type': 'image_url',
                            'image_url': image_url,
                            # 输入图像的最小像素，小于则原比放大直到大于该值
                            'min_pixels': 28 * 28 * 4,
                            # 输入图像的最大像素，大于则原比缩小直到小于该值
                            'max_pixels': 28 * 28 * 8192
                        },
                        {'type': 'text', 'text': prompt}
                    ]
                }
            ],
        )
        result = completion.choices[0].message.content

    except Exception as e:
        print('报错信息：', e)
    return result


# 本地文件
def path_image_cline(prompt, image_path):
    messages = [
        {
            'role': 'user',
            'content': [
                {
                    'image': image_path,
                    'min_pixels': 28 * 28 * 4,
                    'max_pixels': 28 * 28 * 8192,
                    'enable_rotate': True,
                },
                {'text': prompt}
            ]
        }
    ]

    response = MultiModalConversation.call(
        api_key='sk-5070859462a14565a7d30fa7778267b2',
        model='qwen-vl-ocr',
        messages=messages,
    )
    result = response['output']['choices'][0]['message'].content[0]['text']
    return result


def image_text_extract(prompt, image_fil):
    if os.path.isfile(image_fil):
        # print('本地文件')
        result = path_image_cline(prompt, image_fil)
    else:
        # print('图片链接')
        result = url_image_client(prompt, image_fil)
    return result


if __name__ == '__main__':
    prompt = '提取图片中的文字内容，不要输出任何额外信息'
    result = image_text_extract(prompt, 'https://img2.baidu.com/it/u=2341919796,2553569242&fm=253&fmt=auto&app=138&f=JPEG?w=500&h=1004')
    print(result)

