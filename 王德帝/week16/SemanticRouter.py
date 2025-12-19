from typing import Optional, List, Any, Dict, Callable

import faiss
import numpy as np


class SemanticRouter:

    def __init__(
        self,
        embedding_method: Callable[[List[str]], Any],
        distance_threshold: float = 0.3,
        faiss_index_path: Optional[str] = None,
    ):

        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.faiss_index_path = faiss_index_path

        # 语义索引（基于 FAISS），用于快速相似度搜索
        self.index: Optional[faiss.IndexFlatL2] = None

        # 用于从“向量 id”反查业务 target：
        #   - self.targets[i] 代表第 i 条向量对应的 target（例如 "refund"）
        self.targets: List[str] = []

        # 可选：也可以保存每个样本的原始问题文本，方便调试与可视化
        self.samples: List[str] = []

    # =========================
    # 构建“意图库”
    # =========================
    def add_route(self, questions: List[str], target: str) -> None:

        if not questions:
            return

        # 1. 编码问题为向量
        embeddings = self.embedding_method(questions)  # 伪代码：外部传入的 encoder
        embeddings = np.array(embeddings, dtype="float32")
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # 2. 初始化索引（如果还没建） —— 使用 L2 距离
        if self.index is None:
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)

        # 3. 向索引中新增向量
        self.index.add(embeddings)  # 向量个数 = len(questions)

        # 4. 维护向量 id -> target / 原始问题 的映射
        self.targets.extend([target] * len(questions))
        self.samples.extend(questions)

        # 5. 将索引持久化到磁盘，便于后续加载
        # if self.faiss_index_path:
        #     faiss.write_index(self.index, self.faiss_index_path)

    # =========================
    # 意图识别
    # =========================
    def route(self, question: str) -> Optional[str]:

        if self.index is None or not self.targets:
            # 还没有任何路由信息，直接返回 None
            return None

        # 1. 编码问题向量
        query_emb = self.embedding_method([question])  # 统一使用 List[str] 接口
        query_emb = np.array(query_emb, dtype="float32")
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)

        # 2. 检索最近的若干个候选（这里简单取 5 个，可按需调整）
        k = min(5, self.index.ntotal)
        distances, indices = self.index.search(query_emb, k)

        # 3. 取距离最近的那个样本
        best_distance = float(distances[0][0])
        best_index = int(indices[0][0])

        # 4. 如果距离太大，说明语义差异较大，视为“未识别到意图”
        if best_distance > self.distance_threshold:
            return None

        # 5. 返回该向量对应的业务 target
        return self.targets[best_index]


if __name__ == "__main__":
    # 伪代码：一个极简的 embedding 函数，用全 1 向量代替真实向量
    def mock_embedding(texts: List[str]):
        """
        这里只是演示用的假向量：
        - 实际场景请替换为真实的 embedding 模型（如 BGE、text-embedding-3-large 等）
        """
        return np.array([np.ones(8) for _ in texts], dtype="float32")

    router = SemanticRouter(embedding_method=mock_embedding, distance_threshold=0.5)

    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting",
    )
    router.add_route(
        questions=["如何退货"],
        target="refund",
    )

    result = router.route("Hi, good morning")
    print("route result:", result)
