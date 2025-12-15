from fastmcp import FastMCP
import numpy as np
import math
mcp = FastMCP(
    name="Formula-MCP-Server",
    instructions="""This server contains 10 formulas.""",
)

# 00ac792a-04dd-4639-abbd-d7f78cbb7ea.pdf
@mcp.tool()
def nonlinear_interaction_model(x: float, y: float) -> float:
    """
    计算双变量非线性交互函数：fun(x, y) = 2.5*sin(x) + 1.8*cos(y) + 0.3*x*y。
    用于模拟两个输入变量的主效应及其交互效应对目标输出的综合影响。

    Args:
        x (float): 第一个输入变量
        y (float): 第二个输入变量

    Returns:
        float: 综合影响后的输出值

    Args:
        x (float): 第一个输入变量
        y (float): 第二个输入变量

    Returns:
        float: 综合影响后的输出值
    """
    result = 2.5 * math.sin(x) + 1.8 * math.cos(y) + 0.3 * x * y
    return result

# 0a948fc4-b083-44c6-af02-70be51108f7.pdf
@mcp.tool()
def dissolved_oxygen_nonlinear_model(
    t: float,
    a: float = 10.0,
    b: float = 0.1,
    c: float = 2.0,
    d: float = 0.5
) -> float:
    """
    模拟溶解氧浓度随时间变化：DO(t) = a * e^(-b*t) + c * sin(d*t)。
    融合了自然衰减和周期性环境扰动，用于水产养殖水质调控。

    Args:
        t (float): 时间
        a (float): 初始溶解氧释放量
        b (float): 衰减系数
        c (float): 环境扰动振幅
        d (float): 环境扰动频率

    Returns:
        float: t时刻的溶解氧浓度
    """
    do_t = a * math.exp(-b * t) + c * math.sin(d * t)
    return max(0, do_t)  # DO浓度不能为负

# 0a96883f-a7ee-4aa7-b020-efdb1f634d0.pdf
@mcp.tool()
def ecommerce_order_prediction(
    ad_spend_t: float,
    discount_rate_t: float,
    prev_orders_t: float,
    alpha: float = 0.05,
    beta: float = 100.0,
    gamma: float = 0.7
) -> float:
    """
    预测当日订单数量：orders_t = alpha*ad_spend_t + beta*discount_rate_t + gamma*prev_orders_t。
    用于库存管理和营销策略优化。

    Args:
        ad_spend_t (float): 当日广告支出
        discount_rate_t (float): 当日折扣力度（如0.9表示9折）
        prev_orders_t (float): 前一日订单数量
        alpha (float): 0.05，表示广告支出对订单量的敏感系数
        beta (float): 100，表示折扣率对订单量的放大系数
        gamma (float): 0.7，表示前一日订单数量对当前日订单趋势的惯性影响

    Returns:
        float: 预测的当日订单数量
    """
    orders_t = alpha * ad_spend_t + beta * discount_rate_t + gamma * prev_orders_t
    return max(0, orders_t)  # 订单量不能为负

# 0afb9da6-158a-48dd-abfb-dc85846390ff.md
@mcp.tool()
def agricultural_yield_prediction(
    temp: float,
    rainfall: float,
    fertilizer: float,
    sunlight: float,
    soil_quality: float,
    base_yield: float = 5.0
) -> float:
    """
    计算多因子乘法模型预测的作物产量：yield = base_yield * temp_factor * ... * soil_factor。
    用于量化温度、降水、施肥、光照和土壤质量对产量的综合影响。

    Args:
        temp (float): 平均生长温度（℃）
        rainfall (float): 生长期间降水量（mm）
        fertilizer (float): 施肥量（kg/ha）
        sunlight (float): 每日平均光照时长（小时）
        soil_quality (float): 土壤质量指数（0-1）
        base_yield (float): 基础产量水平（吨/公顷），默认5.0

    Returns:
        float: 预测的作物产量（吨/公顷）
    """
    temp_factor = 1.0 - abs(temp - 25) / 25
    rainfall_factor = 1.0 - abs(rainfall - 600) / 600
    fertilizer_factor = 1.0 + fertilizer / 200
    sunlight_factor = 0.8 + (sunlight / 12) * 0.4
    soil_factor = 1.0 + soil_quality

    yield_prediction = base_yield * temp_factor * rainfall_factor * fertilizer_factor * sunlight_factor * soil_factor
    return max(0, yield_prediction)  # 产量不能为负

# 0b9a5088-426e-4c65-8bfb-ef98fc410da.pdf
@mcp.tool()
def crop_yield_model(
    F: float,
    I: float,
    T: float,
    a: float = 0.8,
    b: float = 0.5,
    c: float = 0.01
) -> float:
    """
    计算单位面积作物产量：Y = a*F + b*I - c*T²。
    综合考虑土壤肥力、灌溉量和气温的影响。

    Args:
        F (float): 土壤肥力指数
        I (float): 每周灌溉量（mm/week）
        T (float): 平均气温（℃）
        a, b, c (float): 经验系数，分别反映各因素的贡献程度

    Returns:
        float: 单位面积作物产量（kg/ha）
    """
    yield_value = a * F + b * I - c * T**2
    return max(0, yield_value)  # 产量不能为负


# 0b579473-43f3-4f7d-a45a-312089b766a.pdf
@mcp.tool()
def food_drying_model(
    M0: float,
    k: float,
    T: float
) -> float:
    """
    计算食品在干燥时间T内的累计水分蒸发量：Evaporated(T) = M0 * (T + (e^(-k*T) - 1) / k)。
    用于优化干燥工艺参数，提高生产效率。

    Args:
        M0 (float): 初始水分含量
        k (float): 水分蒸发速率常数
        T (float): 干燥时间

    Returns:
        float: 在时间T内蒸发的总水分量
    """
    evaporated = M0 * (T + (math.exp(-k * T) - 1) / k)
    return evaporated

# 0ba15b17-85d2-4944-9a04-a9bd23c2e3f.pdf
@mcp.tool()
def student_performance_model(
    x1: float,
    x2: float,
    x3: float,
    x4: float,
    w1: float = 0.2,
    w2: float = 0.3,
    w3: float = 0.3,
    w4: float = 0.2,
    alpha: float = 0.1,
    beta: float = 5.0
) -> float:
    """
    通过Sigmoid函数评估学生学习效果：Score = 100 / (1 + e^(-alpha*(w1*x1 + w2*x2 + w3*x3 + w4*x4 - beta)))。
    综合考虑学习时长、出勤率、测验成绩和课堂参与度。

    Args:
        x1 (float): 学习时长（小时）
        x2 (float): 出勤率（百分比）
        x3 (float): 平时测验平均分（百分比）
        x4 (float): 课堂参与度（1~5分）
        w1, w2, w3, w4 (float): 各变量的权重系数
        alpha (float): 控制S型曲线陡峭程度
        beta (float): 控制曲线在横轴上的平移位置

    Returns:
        float: 预测的学习成绩（0-100分）
    """
    linear_combination = w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 - beta
    score = 100 / (1 + math.exp(-alpha * linear_combination))
    return score

# 0bcccdd0-b9a4-4f9b-afc2-d14b4384098.pdf
@mcp.tool()
def differential_equation_model(
    x1_t: float,
    x2_t: float,
    x3_t: float,
    y_t_minus_1: float,
    y_t_minus_2: float,
    a: float = 1.0,
    b: float = 0.5,
    c: float = 0.3,
    d: float = 0.2
) -> float:
    """
    计算多输入差分方程模型：yt = a*x1,t + b*yt-1 + c*yt-2 + d*x2,t*x3,t。
    用于模拟和预测具有时序依赖特性的动态系统。

    Args:
        x1_t, x2_t, x3_t (float): 当前时刻的三个输入变量
        y_t_minus_1, y_t_minus_2 (float): 前两个时间步的输出值
        a, b, c, d (float): 模型参数，用于调节各输入项对输出的影响权重

    Returns:
        float: 当前时刻的输出值 yt
    """
    return a * x1_t + b * y_t_minus_1 + c * y_t_minus_2 + d * x2_t * x3_t

# 0d2f19ba-1875-4057-b804-379367fedec.pdf
@mcp.tool()
def quadratic_model(x: float) -> float:
    """
    计算基于二次函数 y = 2x² + 3x + 1 的输出值。
    适用于具有明确因果关系的系统行为分析与预测。

    Args:
        x (float): 输入变量

    Returns:
        float: 对应的输出结果 y
    """
    return 2 * x**2 + 3 * x + 1

# 0d9d4e8c-51c1-474b-b8e2-8b950e53437.pdf
@mcp.tool()
def cultural_influence_score(
    content_quality: float,
    channels: int,
    engagement: float,
    time: float
) -> float:
    """
    计算文化传播项目的综合影响力得分：Influence = content_quality * channels * engagement * time。
    该模型量化了内容质量、传播广度、受众活跃度和传播周期对整体效果的联合影响。

    Args:
        content_quality (float): 内容质量评分（例如 0-10 分）。
        channels (int): 使用的传播渠道数量。
        engagement (float): 受众参与度指标（例如 0-1 的归一化值或百分比）。
        time (float): 传播持续时间（例如天数或周数）。

    Returns:
        float: 综合影响力得分。
    """
    # 核心乘积公式
    influence = content_quality * channels * engagement * time
    return influence

# 0daef473-e660-4984-be4d-940433aa889.pdf
@mcp.tool()
def logistic_population_growth(
    N_t: float,
    r: float,
    K: float
) -> float:
    """
    计算基于逻辑斯蒂增长模型的下一年牛群数量：N_{t+1} = N_t + r * N_t * (1 - N_t / K)。
    该模型考虑了自然增长率和环境承载能力的限制，适用于中长期种群趋势预测。

    Args:
        N_t (float): 第 t 年的牛群数量。
        r (float): 年增长率（例如 0.1 表示 10%）。
        K (float): 环境承载能力，即系统能支持的最大种群数量。

    Returns:
        float: 第 t+1 年的预测牛群数量 N_{t+1}。
    """
    # 逻辑斯蒂增长公式
    N_t_plus_1 = N_t + r * N_t * (1 - N_t / K)
    # 确保数量不为负
    return max(0, N_t_plus_1)

if __name__ == "__main__":
    mcp.run(transport="sse", port=8900)
