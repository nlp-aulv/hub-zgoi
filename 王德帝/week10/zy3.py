import glob, json, os
from PIL import Image
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

# 获取图片路径
img_paths = glob.glob('./data/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/*.jpg')
img_paths.sort()
img_paths = img_paths[:10]  # 只取前10张图片

# 加载文本标注
validation_annotations = json.load(
    open('./data/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json')
)
validation_annotations_dict = {x['image_id']: x['caption'][0] for x in validation_annotations}

# 获取前10张图片对应的文本描述
img_paths_basenames = [os.path.basename(x) for x in img_paths]
img_captions = [validation_annotations_dict[x] for x in img_paths_basenames]

# 加载模型和预处理器
model = ChineseCLIPModel.from_pretrained("./models/AI-ModelScope/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("./models/AI-ModelScope/chinese-clip-vit-base-patch16")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 图片编码
img_image_feat = []
batch_size = 10  # 修改batch_size为10，确保一次处理完所有图片

for idx in tqdm(range(len(img_captions) // batch_size + 1)):
    imgs = [Image.open(path) for path in img_paths[idx*batch_size: (idx+1)*batch_size]]
    
    if len(imgs) == 0:
        break
    
    inputs = processor(images=imgs, return_tensors="pt")
    with torch.no_grad():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        image_features = model.get_image_features(**inputs)
        image_features = image_features.data.cpu().numpy()
        img_image_feat.append(image_features)

img_image_feat = np.vstack(img_image_feat)
img_image_feat = normalize(img_image_feat)
print("图片特征形状:", img_image_feat.shape)

# 文本编码
img_texts_feat = []
batch_size = 10  # 修改batch_size为10，确保一次处理完所有文本

for idx in tqdm(range(len(img_captions) // batch_size + 1)):
    texts = [text for text in img_captions[idx*batch_size: (idx+1)*batch_size]]

    if len(texts) == 0:
        break
    
    inputs = processor(
        text=texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=77
    )

    with torch.no_grad():
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model.text_model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0]
        text_features = model.text_projection(pooled_output)
        text_features = text_features.cpu().numpy()
        img_texts_feat.append(text_features)

img_texts_feat = np.vstack(img_texts_feat)
img_texts_feat = normalize(img_texts_feat)
print("文本特征形状:", img_texts_feat.shape)

# 计算图文相似度并展示匹配结果
similarities = np.dot(img_image_feat, img_texts_feat.T)

print("\n图文相似度矩阵：")
for i in range(10):
    print(f"\n图片{i+1}与各文本的相似度：")
    for j in range(10):
        print(f"文本{j+1}: {similarities[i][j]:.4f}")

print("\n最佳匹配结果：")
for i in range(10):
    best_match = np.argmax(similarities[i])
    print(f"\n图片{i+1}:")
    print(f"最佳匹配文本{best_match+1}: {img_captions[best_match]}")
