import glob, json, os
from PIL import Image
from tqdm import tqdm_notebook
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt

img_paths = glob.glob('D:/download/ai_challenger_caption_validation_20170910/caption_validation_images_20170910/*.jpg')
img_paths.sort()

img_paths = img_paths[:3000]
img_paths[:2]

validation_annotations = json.load(
    open('D:/download/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json')
)

validation_annotations[:2]

validation_annotations_dict = {x['image_id']: x['caption'][0] for x in validation_annotations}

img_paths_basenames = [os.path.basename(x) for x in img_paths]
img_captions = [validation_annotations_dict[x] for x in img_paths_basenames]

Image.open(img_paths[0]).show()

print(img_captions[0])

from PIL import Image
import requests
from transformers import ChineseCLIPProcessor, ChineseCLIPModel
import torch

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