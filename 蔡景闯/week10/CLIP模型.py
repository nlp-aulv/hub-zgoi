import os

from transformers import ChineseCLIPModel, ChineseCLIPProcessor
from PIL import Image
import torch

# CLIP模型，旨在将文本和图像的编码结果映射到同一个向量空间
# CLIP模型由两部分组成：文本编码器和图像编码器

model = ChineseCLIPModel.from_pretrained("../models/chinese-clip-vit-base-patch16") # 中文clip模型
processor = ChineseCLIPProcessor.from_pretrained("../models/chinese-clip-vit-base-patch16") # 预处理

# 图像编码：图像分块-》图像序列-》图像向量
folder_path = "./images"
image_paths = []
for filename in os.listdir(folder_path):
    # 检查文件是否为图片（这里简单检查扩展名）
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
        # 构建完整文件路径
        image_paths.append(os.path.join(folder_path, filename))
print(image_paths)

with open('./images/label.txt', 'r', encoding='utf-8') as f:
    categories = [line.strip() for line in f.readlines()]
print(categories)

images = processor(images=[Image.open(img) for img in image_paths], return_tensors="pt")
image_encode = model.get_image_features(**images)
# 文本编码：文本序列-》文本向量
text = processor(text=categories, return_tensors="pt",padding=True, truncation=True)
text_encode = model.get_text_features(**text)
# 计算余弦相似度
cos = image_encode @ text_encode.T / (image_encode.norm(dim=-1)[:, None] * text_encode.norm(dim=-1)[None, :])  # 使用 [:, None] 增加维度
max_indices = torch.argmax(cos, dim=1)
predict_categories = [categories[i] for i in max_indices]
real_categories = [os.path.basename(image_paths[i]).split('.')[0] for i in range(len(image_paths))]
for i in range(len(image_paths)):
    print(f"Real: {real_categories[i]}, Predict: {predict_categories[i]}, Similarity: {max_value.values[i]:.4f}")

# Real: 山竹, Predict: 山竹, Similarity: 0.4636
# Real: 榴莲, Predict: 榴莲, Similarity: 0.4809
# Real: 橙子, Predict: 橙子, Similarity: 0.4463
# Real: 猕猴桃, Predict: 猕猴桃, Similarity: 0.4638
# Real: 番石榴, Predict: 番石榴, Similarity: 0.4948
# Real: 芒果, Predict: 芒果, Similarity: 0.4533
# Real: 苹果, Predict: 苹果, Similarity: 0.4557
# Real: 荔枝, Predict: 荔枝, Similarity: 0.4571
# Real: 葡萄, Predict: 葡萄, Similarity: 0.4618
# Real: 香蕉, Predict: 香蕉, Similarity: 0.4631

