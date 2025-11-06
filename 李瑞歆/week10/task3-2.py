import numpy as np
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from tqdm import tqdm_notebook
from transformers import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("D:\model\AI-ModelScope\chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("D:\model\AI-ModelScope\chinese-clip-vit-base-patch16") # 预处理

img_image_feat = []

batch_size = 20
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