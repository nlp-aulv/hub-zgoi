from typing import Optional, List, Union, Any, Dict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

class SemanticRouter:
    def __init__(
            self,
            embedding_model: SentenceTransformer,
            threshold: float = 0.7
        ):
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.routes: List[Dict[str, Any]] = []

    def add_route(self, questions: List[str], target: str):
        embeddings = self.embedding_model.encode(questions)
        self.routes.append({
            "questions": questions,
            "target": target,
            "embeddings": embeddings
        })

    def get_max_similarity(self, query: str):
        query_embedding = self.embedding_model.encode(query)
        for route in self.routes:
            similarities = cosine_similarity(query_embedding, route["embeddings"])
            max_similarity = np.max(similarities)
            if max_similarity >= self.threshold:
                return route["target"], max_similarity
        return None, 0
    
    def route(self, query: str):
        target, similarity = self.get_max_similarity(query)
        if target is not None:
            return target
        else:
            return "unknown"

    def __call__(self, query: str):
        self.route(query)

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

    router("Hi, good morning")