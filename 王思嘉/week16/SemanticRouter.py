from typing import Optional, List, Union, Any, Dict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class SemanticRouter:
    def __init__(self, threshold: float = 0.5):
        """
        初始化语义路由器
        
        Args:
            threshold: 相似度阈值，超过此值才认为匹配成功
        """
        self.routes: Dict[str, List[str]] = {}
        self.questions_map: Dict[str, str] = {}  # 问题到目标的映射
        self.vectorizer = TfidfVectorizer()
        self.threshold = threshold
        self.is_fitted = False

    def add_route(self, questions: List[str], target: str):
        """
        添加路由规则
        
        Args:
            questions: 问题列表
            target: 目标路由名称
        """
        if target not in self.routes:
            self.routes[target] = []
        
        for question in questions:
            self.routes[target].append(question)
            self.questions_map[question] = target
            
        # 每次添加新路由后需要重新训练向量化器
        self.is_fitted = False

    def _fit_vectorizer(self):
        """训练TF-IDF向量化器"""
        all_questions = list(self.questions_map.keys())
        if all_questions:
            self.vectorizer.fit(all_questions)
            self.is_fitted = True

    def route(self, question: str) -> Optional[str]:
        """
        根据输入问题进行路由
        
        Args:
            question: 输入问题
            
        Returns:
            匹配的目标路由，如果没有匹配则返回None
        """
        if not self.routes:
            return None
            
        if not self.is_fitted:
            self._fit_vectorizer()
            
        if not self.is_fitted:
            return None
            
        # 获取所有预定义问题
        all_questions = list(self.questions_map.keys())
        if not all_questions:
            return None
            
        # 向量化输入问题和所有预定义问题
        try:
            question_vec = self.vectorizer.transform([question])
            questions_vec = self.vectorizer.transform(all_questions)
            
            # 计算余弦相似度
            similarities = cosine_similarity(question_vec, questions_vec)[0]
            
            # 找到最相似的问题及其相似度
            max_sim_idx = np.argmax(similarities)
            max_similarity = similarities[max_sim_idx]
            
            # 如果相似度超过阈值，则返回对应的目标
            if max_similarity >= self.threshold:
                matched_question = all_questions[max_sim_idx]
                return self.questions_map[matched_question]
            else:
                return None
                
        except Exception as e:
            print(f"Error during routing: {e}")
            return None

    def __call__(self, question: str) -> Optional[str]:
        """
        使对象可调用
        
        Args:
            question: 输入问题
            
        Returns:
            匹配的目标路由
        """
        return self.route(question)


if __name__ == "__main__":
    router = SemanticRouter()
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting"
    )
    router.add_route(
        questions=["如何退货"],
        target="refund"
    )

    result = router("Hi, good morning")
    print(f"Route result: {result}")  # 应该输出: Route result: greeting
    
    result = router("怎么退货")
    print(f"Route result: {result}")  # 可能输出: Route result: refund (取决于相似度)
    
    result = router("未知问题")
    print(f"Route result: {result}")  # 应该输出: Route result: None
