import os
from openai import OpenAI
from dashscope import MultiModalConversation


# url图像
def url_image_cline(image_url, query):
    client = OpenAI(
        api_key='sk-5070859462a14565a7d30fa7778267b2',
        base_url='https://dashscope.aliyuncs.com/compatible-mode/v1'
    )

    completion = client.chat.completions.create(
        model='qwen3-vl-plus',
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': image_url,
                    },
                    {'type': 'text', 'text': query},
                ],
            },
        ],
    )
    result = completion.choices[0].message.content
    return result


# 本地图像
def path_image_cline(image_path, query):
    messages = [
        {
            'role': 'user',
            'content': [
                {'image': image_path},
                {'text': query},
            ],
        },
    ]

    response = MultiModalConversation.call(
        api_key='sk-5070859462a14565a7d30fa7778267b2',
        model='qwen3-vl-plus',
        messages=messages
    )

    result = response['output']['choices'][0]['message'].content[0]['text']
    return result


def image_recognition(image_fil, query):
    if os.path.isfile(image_fil):
        # print('本地文件')
        result = path_image_cline(image_fil, query)
    else:
        # print('图片链接')
        result = url_image_cline(image_fil, query)
    return result


if __name__ == '__main__':
    out = image_recognition('./gou.png', '图片中的是猫还是狗？')
    print(out)
