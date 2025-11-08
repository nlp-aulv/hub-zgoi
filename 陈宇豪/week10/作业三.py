import glob
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm import tqdm_notebook
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

# 加载本地模型
model = ChineseCLIPModel.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")
# model.load_state_dict(torch.load("D:\model\clip\chinese-clip-vit-base-patch16"))
processor = ChineseCLIPProcessor.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")  # 预处理

model.to(torch.device("cuda"))

img_paths = glob.glob('./img/*.ppg')

print("图像路径:" + img_paths)

validation_annotations = json.load(
    open('./json/imag.json')
)
validation_annotations_dict = {x['image_id']: x['caption'][0] for x in validation_annotations}
img_paths_basenames = [os.path.basename(x) for x in img_paths]
img_captions = [validation_annotations_dict[x] for x in img_paths_basenames]

# 图形编码
img_image_feat = []
# 批量处理
batch_size = 20
for idx in tqdm_notebook(range(len(img_captions) // batch_size + 1)):
    imgs = [Image.open(path) for path in img_paths[idx * batch_size: (idx + 1) * batch_size]]

    if len(imgs) == 0:
        break
    # 图像编码
    inputs = processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        # 获取特征
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.numpy()
        img_image_feat.append(image_features)

    break
# 降维处理
img_image_feat = np.vstack(img_image_feat)
# 正则化
img_image_feat = normalize(img_image_feat)

# 文本编码
img_texts_feat = []

batch_size = 20
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

query_idx = 1

sim_result = np.dot(img_texts_feat[query_idx], img_image_feat.T)  # 矩阵计算
sim_idx = sim_result.argsort()[::-1][1:4]

print('输入文本: ', img_captions[query_idx])

plt.figure(figsize=(10, 5))
plt.subplot(131)
plt.imshow(Image.open(img_paths[sim_idx[0]]))
plt.xticks([])
plt.yticks([])

plt.subplot(132)
plt.imshow(Image.open(img_paths[sim_idx[1]]))
plt.xticks([])
plt.yticks([])

plt.subplot(133)
plt.imshow(Image.open(img_paths[sim_idx[2]]))
plt.xticks([])
plt.yticks([])

plt.xticks([])
plt.yticks([])

query_idx = 1

sim_result = np.dot(img_image_feat[query_idx], img_texts_feat.T)
sim_idx = sim_result.argsort()[::-1][1:4]
