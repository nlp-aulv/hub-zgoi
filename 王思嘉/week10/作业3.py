#这里直接调用的api,自己网上找了10张图，5个种类弄的，大模型都对了
from PIL import Image
import requests
from modelscope import ChineseCLIPProcessor, ChineseCLIPModel

model = ChineseCLIPModel.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")
processor = ChineseCLIPProcessor.from_pretrained("AI-ModelScope/chinese-clip-vit-base-patch16")

# Squirtle, Bulbasaur, Charmander, Pikachu in English
texts = ["苹果", "猫", "狗", "香蕉","桃子"]

# compute image feature  图片特征计算
for i in range(10):
    url = f"图文匹配/{i}.jpg"
    image = Image.open(url)
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)
    image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    # compute text features  文本特征计算
    inputs = processor(text=texts, padding=True, return_tensors="pt")
    text_features = model.get_text_features(**inputs)
    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)  # normalize

    # compute image-text similarity scores  计算图片和文本之间的相似度
    inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1)  # probs: [[1.2686e-03, 5.4499e-02, 6.7968e-04, 9.4355e-01]]
    print(f"判断图片{i}为：{texts[probs.argmax()]}")
