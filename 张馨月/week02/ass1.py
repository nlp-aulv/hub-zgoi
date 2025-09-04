"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
import sys
print("当前Python解释器路径：", sys.executable)
import os
import re
import random
from typing import List, Dict, Tuple
from collections import Counter, defaultdict

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# -----------------------
# 0) 基础与路径
# -----------------------
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("outputs", exist_ok=True)

DATA_PATH = "爬虫-新闻标题.txt"  # 确保与脚本同目录（如果不在，请修改为实际路径）

# -----------------------
# 1) 弱监督：关键词 -> 类别
# -----------------------
LABEL_KEYWORDS: Dict[str, List[str]] = {
    "体育": [
        "梅西","C罗","欧冠","女排","朱婷","足球","篮球","国足",
        "英超","西甲","法甲","中超","武磊","恒大","亚冠","NBA",
        "FIFA","世界杯","里皮","范戴克","尤文","利物浦","阿森纳","内马尔"
    ],
    "科技财经": [
        "iPhone","苹果","华为","小米","谷歌","量子","拼多多","A13","特斯拉","贾跃亭",
        "阿里","马云","美联储","股","债","基金","加息","降息","通胀","上市","融资","亏损","营收","财报"
    ],
    "汽车": [
        "雷克萨斯","奔驰","宝马","奥迪","凯迪拉克","马自达","本田","吉利","比亚迪","红旗",
        "路虎","保时捷","SUV","新车","上市","售价","巡航","发动机","混动","车展"
    ],
    "房产": [
        "楼市","地铁","房贷","租金","买房","小区","回迁","拆迁","地块","公寓","物业","安置","楼盘","宅地"
    ],
    "娱乐": [
        "范冰冰","王菲","李湘","章子怡","阿娇","李小璐","黄渤","向佐","郭碧婷","米雪",
        "泫雅","张柏芝","郑爽","任达华","乔任梁","杨幂","陈赫","刘露","许嵩","王一博"
    ],
    "教育": [
        "高校","大学","学生","老师","考研","中学","幼儿园","清华","武大","博士",
        "军训","课程","校长","学院","教授","博导","录取","宿舍","学费","研究生"
    ],
    "游戏": [
        "魔兽","暴雪","Dota2","英雄联盟","Steam","无主之地","怪物猎人",
        "GTA","玩家","联动","试玩","DLC","赛季","天梯","服务器","封禁","怀旧服"
    ],
    "旅游": [
        "旅行","航班","国家博物馆","香格里拉","敦煌","雁荡山","尼泊尔","海滩","景区",
        "飞行员","比基尼","城堡","步道","拍照","打卡","探寻","潜水"
    ],
}

PRIORITY = list(LABEL_KEYWORDS.keys())  # 匹配优先级（从前到后）

# -----------------------
# 2) 读取与打标签
# -----------------------
def read_titles(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    # 去重
    uniq = list(dict.fromkeys(lines))
    return uniq

def assign_label(title: str) -> str:
    # 统计每个类匹配到的关键词数量，取最多的；若并列则返回空（避免歧义）
    counts = {}
    for cat, kws in LABEL_KEYWORDS.items():
        cnt = sum(1 for kw in kws if kw in title)
        if cnt > 0:
            counts[cat] = cnt
    if not counts:
        return ""
    # 找最大匹配数
    max_cnt = max(counts.values())
    cands = [c for c, v in counts.items() if v == max_cnt]
    if len(cands) == 1:
        return cands[0]
    return ""  # 多类并列，跳过

titles = read_titles(DATA_PATH)
pairs: List[Tuple[str, str]] = []
for t in titles:
    lab = assign_label(t)
    if lab:
        pairs.append((t, lab))

print(f"总标题数：{len(titles)}，匹配到弱标签的样本数：{len(pairs)}")
counter = Counter([lab for _, lab in pairs])
print("各类样本数：", dict(counter))

# 若某些类样本过少，可过滤掉
MIN_PER_CLASS = 20
kept_labels = [c for c, n in counter.items() if n >= MIN_PER_CLASS]
pairs = [(t, lab) for (t, lab) in pairs if lab in kept_labels]
print("保留类别：", kept_labels)

# -----------------------
# 3) 字符级 BOW 表征
# -----------------------
def build_char_vocab(texts: List[str], max_vocab_size: int = 1500) -> Dict[str, int]:
    # 统计字符频次，取 Top-K
    freq = Counter()
    for tx in texts:
        for ch in tx:
            freq[ch] += 1
    # 预留：0=PAD（不用于BOW），1=UNK
    vocab = {"<UNK>": 1}
    for ch, _ in freq.most_common(max_vocab_size - 1):
        if ch not in vocab:
            vocab[ch] = len(vocab) + 0  # UNK=1，占一个位置
    return vocab

def text_to_bow(text: str, vocab: Dict[str, int]) -> np.ndarray:
    vec = np.zeros(len(vocab) + 1, dtype=np.float32)  # 索引从1开始，0闲置
    for ch in text:
        idx = vocab.get(ch, 1)  # UNK=1
        vec[idx] += 1.0
    # 归一化（避免长度差异）
    s = vec.sum()
    if s > 0:
        vec = vec / s
    return vec

texts = [t for (t, _) in pairs]
labels_str = [lab for (_, lab) in pairs]
label2idx = {lab: i for i, lab in enumerate(sorted(set(labels_str)))}
idx2label = {i: lab for lab, i in label2idx.items()}
y_all = np.array([label2idx[lab] for lab in labels_str], dtype=np.int64)

vocab = build_char_vocab(texts, max_vocab_size=1500)
X_all = np.stack([text_to_bow(t, vocab) for t in texts]).astype(np.float32)

# -----------------------
# 4) 训练/验证 划分
# -----------------------
def train_val_split(X, y, val_ratio=0.2):
    n = X.shape[0]
    idxs = np.arange(n)
    np.random.shuffle(idxs)
    n_val = int(n * val_ratio)
    val_idx = idxs[:n_val]
    train_idx = idxs[n_val:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

X_train, y_train, X_val, y_val = train_val_split(X_all, y_all, val_ratio=0.2)

# 转 tensor
X_train_t = torch.from_numpy(X_train).to(device)
y_train_t = torch.from_numpy(y_train).to(device)
X_val_t   = torch.from_numpy(X_val).to(device)
y_val_t   = torch.from_numpy(y_val).to(device)

# -----------------------
# 5) 模型定义（MLP）
# -----------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int], num_classes: int):
        super().__init__()
        layers: List[nn.Module] = []
        last = in_dim
        for h in hidden:
            layers.append(nn.Linear(last, h))
            layers.append(nn.ReLU())
            last = h
        layers.append(nn.Linear(last, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# -----------------------
# 6) 训练循环 + 评估
# -----------------------
def run_training(arch: List[int],
                 epochs: int = 20,
                 lr: float = 1e-3,
                 batch_size: int = 64):
    model = MLP(in_dim=X_train.shape[1], hidden=arch, num_classes=len(label2idx)).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": []}

    n = X_train_t.size(0)
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        train_loss_acc = 0.0
        nb = 0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]

            logits = model(xb)
            loss = loss_fn(logits, yb)

            opt.zero_grad()
            loss.backward()
            opt.step()

            train_loss_acc += loss.item()
            nb += 1

        mean_train_loss = train_loss_acc / max(nb, 1)

        # 验证
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = loss_fn(val_logits, y_val_t).item()

        history["train_loss"].append(mean_train_loss)
        history["val_loss"].append(val_loss)

        # 可选：每轮打印
        print(f"[{arch}] Epoch {ep:03d} | TrainLoss={mean_train_loss:.4f} | ValLoss={val_loss:.4f}")

    # 计算最终准确率（可选）
    def acc_of(logits, y_true):
        pred = logits.argmax(dim=1)
        return (pred == y_true).float().mean().item()

    model.eval()
    with torch.no_grad():
        train_acc = acc_of(model(X_train_t), y_train_t)
        val_acc   = acc_of(model(X_val_t),   y_val_t)

    # 绘图（单图，包含两条曲线）
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 自带
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框
    plt.figure()
    plt.plot(range(1, len(history["train_loss"]) + 1), history["train_loss"], label="Train Loss")
    plt.plot(range(1, len(history["val_loss"]) + 1),   history["val_loss"],   label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title(f"文本分类 Loss 曲线 - 架构 {arch}")
    plt.legend()
    plt.tight_layout()
    fig_name = f"outputs/textcls_{'-'.join(map(str,arch))}_loss.png"
    plt.savefig(fig_name, dpi=150)
    plt.show()

    result = {
        "arch": arch,
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss":   history["val_loss"][-1],
        "train_acc": train_acc,
        "val_acc":   val_acc,
        "fig": fig_name
    }
    return result

# -----------------------
# 7) 多结构对比
# -----------------------
ARCHS = [
    [64],               # 一层
    [128, 64],          # 两层
    [256, 128, 64],     # 三层
    [512, 256, 128, 64] # 四层
]

all_results = []
for arch in ARCHS:
    res = run_training(arch, epochs=20, lr=1e-3, batch_size=64)
    all_results.append(res)

# 汇总打印
print("\n=== 各结构最终结果汇总 ===")
for r in all_results:
    print(
        f"架构={r['arch']} | "
        f"最终TrainLoss={r['final_train_loss']:.4f} | 最终ValLoss={r['final_val_loss']:.4f} | "
        f"TrainAcc={r['train_acc']:.3f} | ValAcc={r['val_acc']:.3f} | 曲线图={r['fig']}"
    )
