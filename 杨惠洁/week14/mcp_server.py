import numpy as np
import math
from typing import List, Dict, Any, Optional, Union
import sympy as sp
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json

#  步骤1: 定义所有模型的元数据
MODEL_DESCRIPTIONS = [
    {
        "id": "model_1",
        "name": "溶解氧浓度模型",
        "description": "在水平养殖系统中，溶解氧浓度随时间变化的非线性模型，包含指数衰减和周期性扰动",
        "formula": "DO(t) = a * exp(-b*t) + c * sin(d*t)",
        "domain": "环境科学、水产养殖",
        "keywords": ["溶解氧", "养殖", "非线性", "指数衰减", "正弦波动", "水质"],
        "parameters": ["t", "a", "b", "c", "d"]
    },
    {
        "id": "model_2",
        "name": "电子商务订单预测模型",
        "description": "基于广告支出、折扣力度和前一日订单量的一阶线性差分方程，用于预测当日订单数量",
        "formula": "orders_t = α * ad_spend_t + β * discount_rate_t + γ * prev_orders_t",
        "domain": "电子商务、商业分析",
        "keywords": ["订单预测", "电子商务", "广告支出", "折扣", "线性差分"],
        "parameters": ["ad_spend_t", "discount_rate_t", "prev_orders_t", "α", "β", "γ"]
    },
    {
        "id": "model_3",
        "name": "文化传播影响力模型",
        "description": "基于内容质量、传播渠道、受众参与度和持续时间的线性乘积模型，评估传播项目的综合影响力",
        "formula": "Influence = content_quality * channels * engagement * time",
        "domain": "文化传播、市场营销",
        "keywords": ["影响力", "传播", "内容质量", "参与度", "线性乘积"],
        "parameters": ["content_quality", "channels", "engagement", "time"]
    },
    {
        "id": "model_4",
        "name": "牛群数量增长模型",
        "description": "基于逻辑增长方程的一阶非线性差分方程，模拟种群在有限资源环境下的增长",
        "formula": "N_{t+1} = N_t + r * N_t * (1 - N_t/K)",
        "domain": "畜牧业、生态学",
        "keywords": ["种群增长", "逻辑增长", "承载能力", "差分方程", "非线性"],
        "parameters": ["N_t", "r", "K", "t"]
    },
    {
        "id": "model_5",
        "name": "多变量动态系统模型",
        "description": "包含三个输入变量和两个历史状态的差分方程，用于多变量动态系统建模",
        "formula": "y_t = a * x1_t + b * y_{t-1} + c * y_{t-2} + d * x2_t * x3_t",
        "domain": "控制系统、时间序列分析",
        "keywords": ["动态系统", "差分方程", "多变量", "时间序列", "反馈"],
        "parameters": ["x1_t", "x2_t", "x3_t", "y_{t-1}", "y_{t-2}", "a", "b", "c", "d"]
    },
    {
        "id": "model_6",
        "name": "学生学习效果评估模型",
        "description": "基于Sigmoid函数的学生学习效果评估模型，综合考虑学习时长、出勤率、测验成绩和课堂参与度",
        "formula": "Score = 100 / (1 + exp(-α * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - β)))",
        "domain": "教育评估、学习分析",
        "keywords": ["学生评估", "学习效果", "Sigmoid", "教育", "评分"],
        "parameters": ["x1", "x2", "x3", "x4", "w1", "w2", "w3", "w4", "α", "β"]
    },
    {
        "id": "model_7",
        "name": "二次函数确定性模型",
        "description": "基于二次函数的确定性模型，用于演示输入与输出之间的明确数学关系",
        "formula": "y = 2x^2 + 3x + 1",
        "domain": "数学建模、系统分析",
        "keywords": ["二次函数", "确定性模型", "数学关系", "抛物线"],
        "parameters": ["x"]
    },
    {
        "id": "model_8",
        "name": "作物产量预测模型",
        "description": "基于土壤肥力、灌溉量和气温的作物产量预测模型",
        "formula": "Y = a * F + b * I - c * T^2",
        "domain": "农业科学、作物生产",
        "keywords": ["作物产量", "农业", "土壤肥力", "灌溉", "温度"],
        "parameters": ["F", "I", "T", "a", "b", "c"]
    },
    {
        "id": "model_9",
        "name": "一阶线性差分方程模型",
        "description": "具有记忆特性的一阶线性差分方程，当前状态依赖于前一状态和五个输入变量",
        "formula": "y_t = a * y_{t-1} + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)",
        "domain": "系统建模、经济预测",
        "keywords": ["一阶差分", "线性方程", "系统状态", "记忆特性", "多输入"],
        "parameters": ["y_{t-1}", "x1_t", "x2_t", "x3_t", "x4_t", "x5_t", "a", "b", "c", "d", "e"]
    },
    {
        "id": "model_10",
        "name": "食品干燥水分蒸发模型",
        "description": "食品干燥过程中水分含量变化和累计蒸发量的预测模型",
        "formula": "M(t) = M0 * exp(-k*t), Evaporated(T) = M0 * (T + (exp(-k*T) - 1)/k)",
        "domain": "食品加工、干燥工艺",
        "keywords": ["食品干燥", "水分蒸发", "指数衰减", "食品加工", "工艺优化"],
        "parameters": ["M0", "k", "t"]
    }
]


#  步骤2: 定义RAG检索器
class ModelRetriever:
    def __init__(self, model_descriptions: List[Dict]):
        self.model_descriptions = model_descriptions
        # 使用轻量级句子编码模型
        self.encoder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self._build_index()

    def _build_index(self):
        """构建模型描述的向量索引"""
        texts = []
        for model in self.model_descriptions:
            # 组合描述、公式和关键词作为检索文本
            text = f"{model['name']} {model['description']} {model['formula']} {' '.join(model['keywords'])}"
            texts.append(text)

        self.embeddings = self.encoder.encode(texts, convert_to_tensor=True)

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """检索与查询最相关的模型"""
        query_embedding = self.encoder.encode([query], convert_to_tensor=True)

        # 计算余弦相似度
        similarities = cosine_similarity(
            query_embedding.cpu().numpy(),
            self.embeddings.cpu().numpy()
        )[0]

        # 获取top_k索引
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            model = self.model_descriptions[idx].copy()
            model['similarity'] = float(similarities[idx])
            results.append(model)

        return results

    def rerank(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """对候选模型进行重排序（基于更细粒度的匹配）"""
        for candidate in candidates:
            # 计算关键词匹配分数
            keywords = candidate.get('keywords', [])
            keyword_score = sum(1 for keyword in keywords if keyword.lower() in query.lower())

            # 计算公式关键词匹配
            formula = candidate.get('formula', '')
            formula_score = 0
            formula_terms = ['exp', 'sin', 'log', 'sqrt', '^', '*', '+', '-', '/']
            for term in formula_terms:
                if term in formula and term in query:
                    formula_score += 1

            # 综合分数
            candidate['rerank_score'] = candidate['similarity'] * 0.7 + keyword_score * 0.2 + formula_score * 0.1

        # 按重排序分数排序
        return sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)


#  步骤3: 定义MCP工具执行器
class MCPExecutor:
    """MCP工具执行器"""

    @staticmethod
    def execute_model_1(**kwargs):
        """溶解氧浓度模型"""
        t = kwargs.get('t', 0)
        a = kwargs.get('a', 1.0)
        b = kwargs.get('b', 0.1)
        c = kwargs.get('c', 0.5)
        d = kwargs.get('d', 1.0)

        if isinstance(t, (list, np.ndarray)):
            t = np.asarray(t)
            result = a * np.exp(-b * t) + c * np.sin(d * t)
        else:
            result = a * math.exp(-b * t) + c * math.sin(d * t)

        return {
            "model": "溶解氧浓度模型",
            "formula": "DO(t) = a * exp(-b*t) + c * sin(d*t)",
            "parameters": kwargs,
            "result": result
        }

    @staticmethod
    def execute_model_2(**kwargs):
        """电子商务订单预测模型"""
        ad_spend = kwargs.get('ad_spend_t', 0)
        discount_rate = kwargs.get('discount_rate_t', 0)
        prev_orders = kwargs.get('prev_orders_t', 0)
        alpha = kwargs.get('α', 0.05)
        beta = kwargs.get('β', 100.0)
        gamma = kwargs.get('γ', 0.7)

        orders = alpha * ad_spend + beta * discount_rate + gamma * prev_orders

        return {
            "model": "电子商务订单预测模型",
            "formula": "orders_t = α * ad_spend_t + β * discount_rate_t + γ * prev_orders_t",
            "parameters": kwargs,
            "result": orders
        }

    @staticmethod
    def execute_model_3(**kwargs):
        """文化传播影响力模型"""
        content_quality = kwargs.get('content_quality', 1.0)
        channels = kwargs.get('channels', 1)
        engagement = kwargs.get('engagement', 1.0)
        time = kwargs.get('time', 1.0)

        influence = content_quality * channels * engagement * time

        return {
            "model": "文化传播影响力模型",
            "formula": "Influence = content_quality * channels * engagement * time",
            "parameters": kwargs,
            "result": influence
        }

    @staticmethod
    def execute_model_4(**kwargs):
        """牛群数量增长模型"""
        N_t = kwargs.get('N_t', 100.0)
        r = kwargs.get('r', 0.3)
        K = kwargs.get('K', 1000.0)
        years = kwargs.get('years', 10)

        population = [N_t]
        current = N_t

        for _ in range(years):
            next_pop = current + r * current * (1 - current / K)
            population.append(next_pop)
            current = next_pop

        return {
            "model": "牛群数量增长模型",
            "formula": "N_{t+1} = N_t + r * N_t * (1 - N_t/K)",
            "parameters": kwargs,
            "result": population
        }

    @staticmethod
    def execute_model_5(**kwargs):
        """多变量动态系统模型"""
        x1_t = kwargs.get('x1_t', 0)
        x2_t = kwargs.get('x2_t', 0)
        x3_t = kwargs.get('x3_t', 0)
        y_t_minus_1 = kwargs.get('y_{t-1}', 0)
        y_t_minus_2 = kwargs.get('y_{t-2}', 0)
        a = kwargs.get('a', 1.0)
        b = kwargs.get('b', 1.0)
        c = kwargs.get('c', 1.0)
        d = kwargs.get('d', 1.0)

        y_t = a * x1_t + b * y_t_minus_1 + c * y_t_minus_2 + d * x2_t * x3_t

        return {
            "model": "多变量动态系统模型",
            "formula": "y_t = a * x1_t + b * y_{t-1} + c * y_{t-2} + d * x2_t * x3_t",
            "parameters": kwargs,
            "result": y_t
        }

    @staticmethod
    def execute_model_6(**kwargs):
        """学生学习效果评估模型"""
        x1 = kwargs.get('x1', 0)  # 学习时长
        x2 = kwargs.get('x2', 0)  # 出勤率
        x3 = kwargs.get('x3', 0)  # 测验平均分
        x4 = kwargs.get('x4', 0)  # 课堂参与度
        w1 = kwargs.get('w1', 0.4)
        w2 = kwargs.get('w2', 0.3)
        w3 = kwargs.get('w3', 0.2)
        w4 = kwargs.get('w4', 0.1)
        alpha = kwargs.get('α', 1.0)
        beta = kwargs.get('β', 0.0)

        # 参与度标准化（1-5分到0-1）
        if 1 <= x4 <= 5:
            x4_normalized = (x4 - 1) / 4
        else:
            x4_normalized = x4

        weighted_sum = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4_normalized
        score = 100 / (1 + math.exp(-alpha * (weighted_sum - beta)))

        return {
            "model": "学生学习效果评估模型",
            "formula": "Score = 100 / (1 + exp(-α * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - β)))",
            "parameters": kwargs,
            "result": score
        }

    @staticmethod
    def execute_model_7(**kwargs):
        """二次函数确定性模型"""
        x = kwargs.get('x', 0)

        if isinstance(x, (list, np.ndarray)):
            x = np.asarray(x)
            y = 2 * x ** 2 + 3 * x + 1
        else:
            y = 2 * x ** 2 + 3 * x + 1

        return {
            "model": "二次函数确定性模型",
            "formula": "y = 2x^2 + 3x + 1",
            "parameters": kwargs,
            "result": y
        }

    @staticmethod
    def execute_model_8(**kwargs):
        """作物产量预测模型"""
        F = kwargs.get('F', 0)  # 土壤肥力
        I = kwargs.get('I', 0)  # 灌溉量
        T = kwargs.get('T', 0)  # 温度
        a = kwargs.get('a', 1.0)
        b = kwargs.get('b', 1.0)
        c = kwargs.get('c', 0.1)

        Y = a * F + b * I - c * T ** 2

        return {
            "model": "作物产量预测模型",
            "formula": "Y = a * F + b * I - c * T^2",
            "parameters": kwargs,
            "result": Y
        }

    @staticmethod
    def execute_model_9(**kwargs):
        """一阶线性差分方程模型"""
        y_prev = kwargs.get('y_{t-1}', 0)
        x1_t = kwargs.get('x1_t', 0)
        x2_t = kwargs.get('x2_t', 0)
        x3_t = kwargs.get('x3_t', 0)
        x4_t = kwargs.get('x4_t', 0)
        x5_t = kwargs.get('x5_t', 0)
        a = kwargs.get('a', 0.5)
        b = kwargs.get('b', 0.2)
        c = kwargs.get('c', 0.1)
        d = kwargs.get('d', 0.15)
        e = kwargs.get('e', 0.05)

        y_t = a * y_prev + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)

        return {
            "model": "一阶线性差分方程模型",
            "formula": "y_t = a * y_{t-1} + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)",
            "parameters": kwargs,
            "result": y_t
        }

    @staticmethod
    def execute_model_10(**kwargs):
        """食品干燥水分蒸发模型"""
        M0 = kwargs.get('M0', 1.0)
        k = kwargs.get('k', 0.1)
        t = kwargs.get('t', 0)

        # 水分含量
        M_t = M0 * math.exp(-k * t)

        # 累计蒸发量
        if k != 0:
            evaporated = M0 * (t + (math.exp(-k * t) - 1) / k)
        else:
            evaporated = 0

        return {
            "model": "食品干燥水分蒸发模型",
            "formula": f"M(t) = M0 * exp(-k*t), Evaporated = M0 * (t + (exp(-k*t) - 1)/k)",
            "parameters": kwargs,
            "result": {
                "moisture_content": M_t,
                "total_evaporated": evaporated,
                "remaining_moisture": M0 - evaporated
            }
        }

    @staticmethod
    def get_executor(model_id: str):
        """获取对应的执行器函数"""
        executors = {
            "model_1": MCPExecutor.execute_model_1,
            "model_2": MCPExecutor.execute_model_2,
            "model_3": MCPExecutor.execute_model_3,
            "model_4": MCPExecutor.execute_model_4,
            "model_5": MCPExecutor.execute_model_5,
            "model_6": MCPExecutor.execute_model_6,
            "model_7": MCPExecutor.execute_model_7,
            "model_8": MCPExecutor.execute_model_8,
            "model_9": MCPExecutor.execute_model_9,
            "model_10": MCPExecutor.execute_model_10
        }
        return executors.get(model_id)


#  步骤4: 定义完整的问答系统
class ModelQASystem:
    def __init__(self):
        self.retriever = ModelRetriever(MODEL_DESCRIPTIONS)
        self.executor = MCPExecutor()

    def parse_query(self, query: str) -> Dict:
        """解析用户查询，提取参数"""
        # 简单的参数提取（实际应用中可以使用更复杂的NLP方法）
        params = {}

        # 提取数字参数
        import re
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", query)

        # 常见参数关键词映射
        param_keywords = {
            '时间': 't',
            '初始': 'a',
            '衰减': 'b',
            '振幅': 'c',
            '频率': 'd',
            '广告': 'ad_spend_t',
            '折扣': 'discount_rate_t',
            '前一天': 'prev_orders_t',
            '质量': 'content_quality',
            '渠道': 'channels',
            '参与': 'engagement',
            '增长率': 'r',
            '承载': 'K',
            '肥力': 'F',
            '灌溉': 'I',
            '温度': 'T',
            '水分': 'M0',
            '速率': 'k'
        }

        for keyword, param_name in param_keywords.items():
            if keyword in query:
                # 尝试找到相关的数值
                for i, num in enumerate(numbers):
                    params[param_name] = float(num)
                    break

        # 如果没有提取到参数，设置默认值
        if not params:
            params = {'default_mode': True}

        return params

    def answer_query(self, query: str) -> Dict:
        """处理用户查询的完整流程"""
        # 步骤1: RAG检索相关模型
        candidates = self.retriever.retrieve(query, top_k=3)

        # 步骤2: 重排序
        reranked = self.retriever.rerank(query, candidates)

        # 步骤3: 选择最佳模型
        best_model = reranked[0]

        # 步骤4: 解析参数
        params = self.parse_query(query)

        # 步骤5: 执行模型
        executor_func = self.executor.get_executor(best_model['id'])
        if executor_func:
            result = executor_func(**params)

            # 步骤6: 生成回答
            response = {
                "selected_model": best_model['name'],
                "model_description": best_model['description'],
                "model_formula": best_model['formula'],
                "retrieval_score": best_model['similarity'],
                "parameters_used": params,
                "calculation_result": result['result'],
                "alternative_models": [
                    {
                        "name": m['name'],
                        "similarity": m['similarity']
                    }
                    for m in reranked[1:3]
                ]
            }

            return response
        else:
            return {
                "error": "未找到合适的模型执行器",
                "selected_model": best_model['name'],
                "suggestions": [m['name'] for m in reranked[1:5]]
            }


# 步骤5: 测试系统
def test_system():
    """测试问答系统"""
    system = ModelQASystem()

    # 测试查询
    test_queries = [
        "预测广告支出5000元，折扣率15%，前一天订单300单的情况下的当日订单量",
        "计算土壤肥力75，每周灌溉50mm，平均气温25℃时的作物产量",
        "学生学习50小时，出勤率90%，测验平均分85，课堂参与度4分，预测他的得分",
        "初始水分含量0.8，蒸发速率0.05，干燥10小时后的水分蒸发量",
        "计算x=2时的二次函数值"
    ]

    for query in test_queries:
        print(f"\n{'=' * 60}")
        print(f"查询: {query}")
        print(f"{'=' * 60}")

        response = system.answer_query(query)

        print(f"选中的模型: {response.get('selected_model', 'N/A')}")
        print(f"公式: {response.get('model_formula', 'N/A')}")
        print(f"计算结果: {response.get('calculation_result', 'N/A')}")

        if 'alternative_models' in response:
            print(f"\n备选模型:")
            for alt in response['alternative_models']:
                print(f"  - {alt['name']} (相似度: {alt['similarity']:.3f})")


# 步骤6: 提供使用接口
def interactive_query():
    """交互式查询接口"""
    system = ModelQASystem()

    print("数学模型问答系统已启动！")
    print("支持查询的模型包括：溶解氧模型、订单预测、传播影响力、种群增长等10个模型")
    print("输入 'quit' 或 'exit' 退出")
    print("-" * 60)

    while True:
        query = input("\n请输入您的问题: ").strip()

        if query.lower() in ['quit', 'exit', 'q']:
            print("感谢使用，再见！")
            break

        if not query:
            continue

        try:
            response = system.answer_query(query)

            print(f"\n选中模型: {response.get('selected_model', '未知模型')}")
            print(f"模型描述: {response.get('model_description', '')}")
            print(f"使用公式: {response.get('model_formula', '')}")
            print(f"计算结果: {response.get('calculation_result', '')}")

            if 'error' in response:
                print(f"警告: {response['error']}")

        except Exception as e:
            print("请尝试重新表述您的问题。")


if __name__ == "__main__":

    # 测试系统
    test_system()