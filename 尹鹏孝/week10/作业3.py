# 作业3: 加载中文的clip模型，只要cpu推理，跑完 01-CLIP模型.ipynb
# https://www.modelscope.cn/models/AI-ModelScope/chinese-clip-vit-base-patch16
# 可以不用原始数据，任意10个图 10个文本，完成图文匹配。

import glob, json, os
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch
import jieba
img_paths = glob.glob('./img/3/*.jpg')
img_paths.sort()

img_paths = img_paths[:10]
img_paths[:10]

validation_annotations = json.load(
    open('./data/3.json',encoding="utf-8")
)

validation_annotations_dict = {os.path.basename(x['url']): x['caption'][0] for x in validation_annotations['data']}

img_paths_basenames = [os.path.basename(x) for x in img_paths]
img_captions = [validation_annotations_dict[x] for x in img_paths_basenames]

Image.open(img_paths[0])

# print(validation_annotations['data'][:2])



model = ChineseCLIPModel.from_pretrained("../model/AI-ModelScope/chinese-clip-vit-base-patch16/",weights_only=False) # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("../model/AI-ModelScope/chinese-clip-vit-base-patch16/",weights_only=False) # 预处理

img_image_feat = []

batch_size = 10
for idx in tqdm_notebook(range(len(img_captions) // batch_size + 1)):
    imgs = [Image.open(path) for path in img_paths[idx * batch_size: (idx + 1) * batch_size]]

    if len(imgs) == 0:
        break

    inputs = processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.numpy()
        img_image_feat.append(image_features)

    break

img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)


# img_image_feat.shape # 10张图片 512 维度
print(img_image_feat.shape)

img_texts_feat = []

batch_size = 10
for idx in tqdm_notebook(range(len(img_captions) // batch_size + 1)):
    texts = [text for text in img_captions[idx * batch_size: (idx + 1) * batch_size]]

    if len(texts) == 0:
        break

    inputs = processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
        img_texts_feat.append(text_features)
    break

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)

print(img_texts_feat.shape)




query_idx = 9

sim_result = np.dot(img_texts_feat[query_idx], img_image_feat.T) # 矩阵计算
sim_idx = sim_result.argsort()[::-1][1:4]
#%%
print('输入文本: ', img_captions[query_idx])

plt.figure(figsize=(9, 5))
plt.subplot(131)
plt.imshow(Image.open(img_paths[sim_idx[0]]))
plt.xticks([]); plt.yticks([])

plt.subplot(132)
plt.imshow(Image.open(img_paths[sim_idx[1]]))
plt.xticks([]); plt.yticks([])

plt.subplot(133)
plt.imshow(Image.open(img_paths[sim_idx[2]]))
plt.xticks([]); plt.yticks([])



query_idx = 9

sim_result = np.dot(img_image_feat[query_idx], img_texts_feat.T)
sim_idx = sim_result.argsort()[::-1][1:4]
#%%
plt.imshow(Image.open(img_paths[query_idx]))

print('文本识别结果: ', [img_captions[x] for x in sim_idx])


jieba.lcut('今天天气很好，心情也很好。')

jieba.lcut(img_captions[0])


img_captions2words = [jieba.lcut(x) for x in img_captions]
img_captions2words = sum(img_captions2words, [])

img_captions2words[:9]

img_texts_feat = []

batch_size = 10
for idx in tqdm_notebook(range(len(img_captions2words) // batch_size + 1)):
    texts = [text for text in img_captions2words[idx * batch_size: (idx + 1) * batch_size]]

    if len(texts) == 0:
        break

    inputs = processor(text=texts, return_tensors="pt", padding=True)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features = text_features.data.numpy()
        img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)


print(img_texts_feat.shape)


query_idx = 9

sim_result = np.dot(img_image_feat[query_idx], img_texts_feat.T)
sim_idx = sim_result.argsort()[::-1][1:7]
#%%
plt.imshow(Image.open(img_paths[query_idx]))

print('文本识别结果: ', [img_captions2words[x] for x in sim_idx])