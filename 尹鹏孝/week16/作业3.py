from typing import Optional, List, Union, Any, Dict, Callable
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import os


class SemanticRouter:
    def __init__(
            self,
            embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2",
            similarity_threshold: float = 0.7,
            cache_dir: str = "./semantic_router_cache"
    ):
        """
        初始化语义路由器

        Args:
            embedding_model: 使用的嵌入模型名称
            similarity_threshold: 相似度阈值，高于此值才认为匹配
            cache_dir: 缓存目录，用于保存FAISS索引
        """
        self.model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.cache_dir = cache_dir

        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)

        # 存储路由信息
        self.routes = {}  # target -> {"questions": [], "embeddings": np.array, "count": int}
        self.route_embeddings = []  # 所有问题的嵌入
        self.route_targets = []  # 对应的目标
        self.index = None
        self._is_index_initialized = False

    def _get_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """获取文本的嵌入向量"""
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings

    def _initialize_index(self, embedding_dim: int):
        """初始化FAISS索引"""
        self.index = faiss.IndexFlatIP(embedding_dim)  # 使用内积（余弦相似度）
        self._is_index_initialized = True

    def _save_index(self, target: str):
        """保存索引到文件"""
        if self.index is not None:
            index_path = os.path.join(self.cache_dir, f"{target}_index.bin")
            faiss.write_index(self.index, index_path)

    def _load_index(self, target: str) -> bool:
        """从文件加载索引"""
        index_path = os.path.join(self.cache_dir, f"{target}_index.bin")
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
            self._is_index_initialized = True
            return True
        return False

    def add_route(self, questions: List[str], target: str, save_to_disk: bool = True):
        """
        添加新的路由规则

        Args:
            questions: 属于该路由的问题示例
            target: 路由目标（标签）
            save_to_disk: 是否保存到磁盘
        """
        if not questions:
            raise ValueError("questions列表不能为空")

        # 获取问题的嵌入
        question_embeddings = self._get_embeddings(questions)

        # 归一化嵌入向量（用于余弦相似度计算）
        faiss.normalize_L2(question_embeddings)

        # 初始化或更新FAISS索引
        if not self._is_index_initialized:
            self._initialize_index(question_embeddings.shape[1])

        # 添加嵌入到索引
        self.index.add(question_embeddings)

        # 更新路由信息
        if target not in self.routes:
            self.routes[target] = {
                "questions": [],
                "embeddings": [],
                "count": 0
            }

        # 记录问题和嵌入
        self.routes[target]["questions"].extend(questions)
        self.routes[target]["embeddings"].append(question_embeddings)
        self.routes[target]["count"] += len(questions)

        # 更新全局列表
        self.route_embeddings.extend(question_embeddings)
        self.route_targets.extend([target] * len(questions))

        # 保存到磁盘
        if save_to_disk:
            self._save_index(target)

            # 保存路由元数据
            meta_path = os.path.join(self.cache_dir, f"{target}_meta.npz")
            np.savez_compressed(
                meta_path,
                questions=questions,
                target=target,
                count=len(questions)
            )

        print(f"路由 '{target}' 已添加，包含 {len(questions)} 个示例问题")

    def route(self, question: str, k: int = 3) -> Optional[Dict[str, Any]]:
        """
        对输入问题进行路由

        Args:
            question: 输入的问题文本
            k: 搜索最近邻的数量

        Returns:
            匹配的路由信息，如果未匹配则返回None
        """
        if not self._is_index_initialized or self.index.ntotal == 0:
            return None

        # 获取问题的嵌入
        query_embedding = self._get_embeddings(question)

        # 归一化
        faiss.normalize_L2(query_embedding)

        # 搜索最相似的k个问题
        similarities, indices = self.index.search(query_embedding, k)

        # 获取最佳匹配
        best_similarity = similarities[0][0]
        best_index = indices[0][0]

        # 检查是否超过阈值
        if best_similarity < self.similarity_threshold or best_index == -1:
            return None

        # 获取对应的目标
        best_target = self.route_targets[best_index]

        # 获取相似的问题示例
        similar_questions = []
        for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx != -1 and sim >= self.similarity_threshold:
                target = self.route_targets[idx]
                question_text = None

                # 在对应的路由中查找具体问题
                if target in self.routes:
                    route_info = self.routes[target]
                    # 计算在路由内的相对索引
                    route_start_idx = sum(
                        self.routes[t]["count"] for t in self.routes
                        if list(self.routes.keys()).index(t) < list(self.routes.keys()).index(target)
                    )
                    relative_idx = idx - route_start_idx

                    if 0 <= relative_idx < len(route_info["questions"]):
                        question_text = route_info["questions"][relative_idx]

                similar_questions.append({
                    "question": question_text,
                    "similarity": float(sim),
                    "target": target
                })

        return {
            "target": best_target,
            "similarity": float(best_similarity),
            "matched_question": self.routes[best_target]["questions"][0] if self.routes[best_target][
                "questions"] else None,
            "confidence": min(1.0, best_similarity / self.similarity_threshold),  # 置信度
            "similar_questions": similar_questions,
            "all_matches": [
                {
                    "target": self.route_targets[idx],
                    "similarity": float(sim)
                }
                for sim, idx in zip(similarities[0], indices[0])
                if idx != -1
            ]
        }

    def __call__(self, question: str) -> Optional[str]:
        """简化调用接口，只返回目标标签"""
        result = self.route(question)
        return result["target"] if result else None

    def get_all_routes(self) -> Dict[str, List[str]]:
        """获取所有路由及其问题示例"""
        return {
            target: info["questions"]
            for target, info in self.routes.items()
        }

    def remove_route(self, target: str):
        """移除指定路由"""
        if target in self.routes:
            del self.routes[target]

            # 重建索引
            self._rebuild_index()

            # 删除缓存文件
            index_path = os.path.join(self.cache_dir, f"{target}_index.bin")
            meta_path = os.path.join(self.cache_dir, f"{target}_meta.npz")

            if os.path.exists(index_path):
                os.remove(index_path)
            if os.path.exists(meta_path):
                os.remove(meta_path)

            print(f"路由 '{target}' 已移除")

    def _rebuild_index(self):
        """重建FAISS索引"""
        if not self.routes:
            self.index = None
            self._is_index_initialized = False
            self.route_embeddings = []
            self.route_targets = []
            return

        # 重新收集所有嵌入
        all_embeddings = []
        all_targets = []

        for target, info in self.routes.items():
            for emb_array in info["embeddings"]:
                all_embeddings.append(emb_array)
                all_targets.extend([target] * len(emb_array))

        if all_embeddings:
            # 合并所有嵌入
            combined_embeddings = np.vstack(all_embeddings)
            faiss.normalize_L2(combined_embeddings)

            # 重新初始化索引
            embedding_dim = combined_embeddings.shape[1]
            self._initialize_index(embedding_dim)
            self.index.add(combined_embeddings)

            # 更新列表
            self.route_embeddings = combined_embeddings.tolist()
            self.route_targets = all_targets


if __name__ == "__main__":
    # 示例用法
    router = SemanticRouter(
        embedding_model="paraphrase-multilingual-MiniLM-L12-v2",
        similarity_threshold=0.7
    )

    # 添加路由
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon", "Hello", "Hey there"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货", "怎么退款", "退货流程", "退款申请"],
        target="refund"
    )
    router.add_route(
        questions=["What is the weather?", "天气怎么样", "今天天气如何"],
        target="weather"
    )

    # 测试路由
    test_questions = [
        "Hi, good morning",
        "How are you doing?",
        "如何退货",
        "我想退款",
        "What's the weather like today?",
        "This is an unrelated question"
    ]

    for question in test_questions:
        result = router.route(question)
        if result:
            print(f"问题: '{question}'")
            print(f"  路由到: {result['target']} (置信度: {result['confidence']:.2f})")
            print(f"  相似度: {result['similarity']:.3f}")
            print(f"  匹配的示例: {result['matched_question']}")
            print()
        else:
            print(f"问题: '{question}' - 未找到匹配的路由")
            print()

    # 简化调用
    print("简化调用结果:")
    print(f"'Hi, good afternoon' -> {router('Hi, good afternoon')}")
    print(f"'如何退货' -> {router('如何退货')}")
    print(f"'Unknown question' -> {router('Unknown question')}")

    # 查看所有路由
    print("\n所有路由:")
    for target, questions in router.get_all_routes().items():
        print(f"{target}: {questions}")