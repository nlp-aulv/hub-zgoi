import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import random

# 设置随机种子以确保结果可重现
np.random.seed(42)
random.seed(42)


# 1. 创建一个有逻辑意义的学生社交网络
def create_student_network():
    G = nx.Graph()
    # 创建三个社团：技术社、艺术社和体育社
    tech_club = [f"Tech_Student_{i}" for i in range(15)]
    art_club = [f"Art_Student_{i}" for i in range(15)]
    sports_club = [f"Sports_Student_{i}" for i in range(15)]

    # 添加所有节点
    for node in tech_club + art_club + sports_club:
        G.add_node(node)

    # 创建社团内部的紧密连接
    for club in [tech_club, art_club, sports_club]:
        for i, student1 in enumerate(club):
            for j in range(i + 1, len(club)):
                if np.random.random() < 0.6:  # 60%的概率连接
                    G.add_edge(student1, club[j])

    # 添加一些跨社团的连接（桥接学生）
    # 技术社的一些学生也加入艺术社
    for tech_student in tech_club[:5]:
        for art_student in art_club[:3]:
            if np.random.random() < 0.2:
                G.add_edge(tech_student, art_student)

    # 艺术社的一些学生也加入体育社
    for art_student in art_club[5:10]:
        for sports_student in sports_club[5:10]:
            if np.random.random() < 0.2:
                G.add_edge(art_student, sports_student)

    # 体育社的一些学生也加入技术社
    for sports_student in sports_club[10:]:
        for tech_student in tech_club[10:]:
            if np.random.random() < 0.2:
                G.add_edge(sports_student, tech_student)

    return G, tech_club, art_club, sports_club


# 2. 生成随机游走序列（DeepWalk的核心）
def generate_random_walks(G, num_walks, walk_length):
    walks = []
    nodes = list(G.nodes())
    for _ in range(num_walks):
        random.shuffle(nodes)  # 随机打乱节点顺序，确保每个节点都有机会成为游走的起点
        for node in nodes:
            walk = [node]
            while len(walk) < walk_length:
                cur = walk[-1]  # 当前节点
                neighbors = list(G.neighbors(cur))  # 获取其邻居
                if len(neighbors) > 0:
                    walk.append(random.choice(neighbors))  # 随机选择一个邻居作为下一个节点
                else:
                    break
            walks.append([str(x) for x in walk])
    return walks


# 3. 使用Word2Vec学习节点表示
def learn_embeddings(walks, dimensions=64, window_size=5):
    # Word2Vec模型接收节点序列作为“句子”
    model = Word2Vec(
        walks,
        vector_size=dimensions,  # 嵌入向量的维度
        window=window_size,  # 窗口大小，用于上下文预测
        min_count=0,
        sg=1,  # 使用skip-gram模型
        workers=4,
        epochs=10
    )
    return model


# 4. 计算节点相似度矩阵
def calculate_similarity_matrix(model, nodes):
    # 获取所有节点的嵌入向量
    embeddings = np.array([model.wv[node] for node in nodes])
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)
    return similarity_matrix


# 5. 可视化函数
def visualize_results(G, tech_club, art_club, sports_club, similarity_matrix, model, top_n=5):
    # 根据社团设置节点颜色
    node_colors = []
    for node in G.nodes():
        if node in tech_club:
            node_colors.append('purple')
        elif node in art_club:
            node_colors.append('orange')
        else:
            node_colors.append('teal')

    plt.figure(figsize=(18, 6))

    # 子图1: 原始网络结构
    plt.subplot(131)
    pos = nx.spring_layout(G, seed=42)  # 使用spring布局
    nx.draw(G, pos, node_color=node_colors, node_size=50, with_labels=False)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='purple', label='Tech'),
        Patch(facecolor='orange', label='Art'),
        Patch(facecolor='teal', label='Sport')
    ]
    plt.legend(handles=legend_elements, loc='best')

    # 子图2: 节点嵌入的2D投影
    plt.subplot(132)
    all_nodes = list(G.nodes())
    embeddings = np.array([model.wv[node] for node in all_nodes])
    pca = PCA(n_components=2)  # 使用PCA将高维嵌入降到2维
    embeddings_2d = pca.fit_transform(embeddings)

    for i, node in enumerate(all_nodes):
        color = 'purple' if node in tech_club else 'orange' if node in art_club else 'teal'
        plt.scatter(embeddings_2d[i, 0], embeddings_2d[i, 1], c=color, s=100)

    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend(handles=legend_elements, loc='best')

    # 子图3: 节点相似度热力图
    plt.subplot(133)
    # 选取部分节点展示热力图
    selected_nodes = tech_club[:3] + art_club[:3] + sports_club[:3]
    selected_indices = [all_nodes.index(node) for node in selected_nodes]
    selected_similarity = similarity_matrix[np.ix_(selected_indices, selected_indices)]

    plt.imshow(selected_similarity, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Cosine Similarity')
    plt.xticks(range(len(selected_nodes)), [s.replace('Student_', '') for s in selected_nodes], rotation=45, ha='right')
    plt.yticks(range(len(selected_nodes)), [s.replace('Student_', '') for s in selected_nodes])
    plt.title("Heatpmap of node Similarity")

    plt.tight_layout()
    plt.show()

    # 打印最相似的节点对
    print("\n最相似的5个节点对:")
    similarities = []
    for i in range(len(all_nodes)):
        for j in range(i + 1, len(all_nodes)):
            similarities.append((all_nodes[i], all_nodes[j], similarity_matrix[i, j]))
    similarities.sort(key=lambda x: x[2], reverse=True)
    for i in range(min(top_n, len(similarities))):
        node1, node2, sim = similarities[i]
        print(f"{node1} - {node2}: {sim:.4f}")


# 主程序
if __name__ == "__main__":
    G, tech_club, art_club, sports_club = create_student_network()
    print(f"创建学生网络，包含 {G.number_of_nodes()} 个节点和 {G.number_of_edges()} 条边")

    walks = generate_random_walks(G, num_walks=10, walk_length=20)
    print(f"生成了 {len(walks)} 条随机游走序列")

    model = learn_embeddings(walks, dimensions=64, window_size=5)
    print("节点嵌入学习完成")

    all_nodes = list(G.nodes())
    similarity_matrix = calculate_similarity_matrix(model, all_nodes)
    visualize_results(G, tech_club, art_club, sports_club, similarity_matrix, model)

"""
DeepWalk的主要功能是将网络转化为一种“语言”。它通过在图上执行一系列随机游走来实现这一点，从而生成节点序列。在这个类比中，整个网络是“语料库”，每个随机游走都是一个“句子”，而每个节点则是这些句子中的“单词”。

Word2Vec是处理DeepWalk创建的“语言”的机器学习模型。它接收这些节点序列（“句子”），并为每个节点（“单词”）学习一个向量表示（“嵌入”）。这个过程基于一个核心思想：在随机游走中经常出现在一起的节点（即结构相似或经常一起被访问）应该具有相似的向量表示。
"""