import numpy as np
import math
from typing import Union, List, Tuple, Optional


def calculate_dissolved_oxygen(
        t: Union[float, List[float], np.ndarray],
        a: float,
        b: float,
        c: float,
        d: float
) -> Union[float, np.ndarray]:
    """
    在水平养殖系统中，溶解氧（Dissolved Oxygen, DO）是影响水生生物健康和生长的关键环境因子之一。
    为了更好地理解和预测DO的动态变化趋势，建立一个能够反映其非线性行为的数学模型具有重要意义。
    该模型可用于模拟封闭或半封闭养殖系统中DO浓度随时间演变的过程，为水质调控和管理提供理论支持。

    本函数基于非线性动力学建模思路，计算溶解氧浓度随时间变化的规律：
    DO(t) = a * exp(-b*t) + c * sin(d*t)

    该模型融合了指数衰减项和周期性扰动项，分别反映溶解氧的自然消耗过程和环境因素引起的波动特性。

    :param t: 时间点或时间序列，单位为时间单位（如小时）
    :param a: 初始溶解氧释放量，反映系统初始状态下的氧含量
    :param b: 溶解氧的衰减系数，刻画其随时间自然下降的速度
    :param c: 环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度
    :param d: 环境扰动的频率，反映扰动周期的快慢
    :return: 溶解氧浓度值或序列，单位与参数a一致（通常为mg/L）
    """
    # 将输入转换为numpy数组以便向量化计算
    t_array = np.asarray(t)

    # 计算溶解氧浓度
    do_values = a * np.exp(-b * t_array) + c * np.sin(d * t_array)

    # 如果输入是标量，则返回标量；否则返回数组
    if np.isscalar(t):
        return float(do_values)
    return do_values


def predict_daily_orders(
        ad_spend_t: float,
        discount_rate_t: float,
        prev_orders_t: float,
        alpha: Optional[float] = 0.05,
        beta: Optional[float] = 100.0,
        gamma: Optional[float] = 0.7
) -> float:
    """
    在电子商务运营中，准确预测每日订单增长量对于库存管理、资源配置和营销策略优化具有重要意义。
    本函数基于一阶线性差分方程模型，从三个关键业务驱动因素预测当日订单数量：
    广告支出、折扣力度和前一日订单量。

    模型公式：orders_t = alpha * ad_spend_t + beta * discount_rate_t + gamma * prev_orders_t

    其中各参数代表：
    - alpha: 广告支出对订单量的敏感系数，默认值0.05
    - beta: 折扣率对订单量的放大系数，默认值100
    - gamma: 前一日订单数量对当前日订单趋势的惯性影响系数，默认值0.7

    该模型具有良好的可解释性，适用于短期订单趋势模拟与敏感性分析。

    :param ad_spend_t: 当日的广告支出，单位为货币单位
    :param discount_rate_t: 当日的折扣力度，通常为折扣率（如0.1表示10%折扣）
    :param prev_orders_t: 前一日的订单数量，单位为单
    :param alpha: 广告支出敏感系数，默认0.05
    :param beta: 折扣率放大系数，默认100
    :param gamma: 订单惯性系数，默认0.7
    :return: 预测的当日订单数量，单位为单
    """
    # 计算当日订单预测值
    orders_t = alpha * ad_spend_t + beta * discount_rate_t + gamma * prev_orders_t

    # 订单数量应为非负整数，这里根据实际需求可以四舍五入
    # 注意：模型输出可能为浮点数，实际应用中可能需要取整
    return orders_t


def calculate_cultural_influence(
        content_quality: float,
        channels: int,
        engagement: float,
        time_period: float
) -> float:
    """
    在文化传播领域，评估一个传播项目的综合影响力是制定传播策略和优化资源配置的重要依据。
    本模型以内容质量、传播渠道数量、受众参与度以及传播持续时间作为核心输入变量，
    通过线性乘积关系计算综合影响力。

    模型公式：Influence = content_quality * channels * engagement * time

    :param content_quality: 内容质量评分（通常为0-10或0-1的标准化值）
    :param channels: 传播渠道数量（个）
    :param engagement: 受众参与度（通常为0-1的百分比或参与指数）
    :param time_period: 传播持续时间（天、周或月）
    :return: 综合影响力指标
    """
    return content_quality * channels * engagement * time_period


def predict_cattle_population(
        initial_population: float,
        growth_rate: float,
        carrying_capacity: float,
        years: int
) -> List[float]:
    """
    在畜牧业管理中，理解与预测牛群数量的动态变化具有重要意义。
    本模型采用逻辑增长模型（Logistic Growth Model）模拟种群在有限资源环境下的增长特性。

    模型公式：N_{t+1} = N_t + r * N_t * (1 - N_t/K)

    :param initial_population: 初始牛群数量
    :param growth_rate: 年增长率r
    :param carrying_capacity: 环境承载能力K
    :param years: 预测年数
    :return: 各年份的牛群数量列表
    """
    population_series = [initial_population]
    current_population = initial_population

    for _ in range(years):
        next_population = current_population + growth_rate * current_population * (
                    1 - current_population / carrying_capacity)
        population_series.append(next_population)
        current_population = next_population

    return population_series


def calculate_dynamic_system_output(
        x1_t: float,
        x2_t: float,
        x3_t: float,
        y_t_minus_1: float,
        y_t_minus_2: float,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 1.0,
        d: float = 1.0
) -> float:
    """
    在系统建模与时间序列预测任务中，差分方程是一种常用工具。
    本模型构建了一个包含三个输入变量的差分方程，用于模拟并预测系统的下一状态。

    模型公式：y_t = a * x1_t + b * y_{t-1} + c * y_{t-2} + d * x2_t * x3_t

    :param x1_t: 当前时刻的第一个输入变量
    :param x2_t: 当前时刻的第二个输入变量
    :param x3_t: 当前时刻的第三个输入变量
    :param y_t_minus_1: 前一个时间步的输出值
    :param y_t_minus_2: 前两个时间步的输出值
    :param a: 输入变量x1的权重系数
    :param b: 滞后项y_{t-1}的权重系数
    :param c: 滞后项y_{t-2}的权重系数
    :param d: 交互项x2*x3的权重系数
    :return: 当前时刻的输出值y_t
    """
    return a * x1_t + b * y_t_minus_1 + c * y_t_minus_2 + d * x2_t * x3_t


def predict_student_score(
        study_hours: float,
        attendance_rate: float,
        quiz_average: float,
        class_participation: float,
        w1: float = 0.4,
        w2: float = 0.3,
        w3: float = 0.2,
        w4: float = 0.1,
        alpha: float = 1.0,
        beta: float = 0.0
) -> float:
    """
    在教育培训领域，评估学生的学习效果是衡量教学质量和课程成效的重要环节。
    本模型基于Sigmoid函数，综合考虑学习时长、出勤率、平时测验成绩和课堂参与度，
    预测学生的最终得分。

    模型公式：Score = 100 / (1 + exp(-alpha * (w1*x1 + w2*x2 + w3*x3 + w4*x4 - beta)))

    :param study_hours: 学习时长（小时）
    :param attendance_rate: 出勤率（0-1之间的百分比）
    :param quiz_average: 平时测验平均分（0-1之间的百分比）
    :param class_participation: 课堂参与度（1-5分）
    :param w1: 学习时长的权重系数，默认0.4
    :param w2: 出勤率的权重系数，默认0.3
    :param w3: 测验平均分的权重系数，默认0.2
    :param w4: 课堂参与度的权重系数，默认0.1
    :param alpha: Sigmoid函数的陡峭程度参数，默认1.0
    :param beta: Sigmoid函数的平移参数，默认0.0
    :return: 预测的学生得分（0-100分）
    """
    # 将课堂参与度从1-5分映射到0-1范围
    participation_normalized = (class_participation - 1) / 4

    # 计算加权线性组合
    weighted_sum = w1 * study_hours + w2 * attendance_rate + w3 * quiz_average + w4 * participation_normalized

    # 应用Sigmoid函数并缩放到0-100分
    score = 100 / (1 + math.exp(-alpha * (weighted_sum - beta)))

    return score


def quadratic_deterministic_model(
        x: Union[float, List[float], np.ndarray]
) -> Union[float, np.ndarray]:
    """
    在系统行为分析与预测任务中，常常采用确定性模型来描述输入与输出之间的明确关系。
    本模型基于一个预定义的二次函数关系，用于演示如何通过函数表达式对输入变量进行系统性映射。

    模型公式：y = 2x^2 + 3x + 1

    :param x: 输入变量或输入数组
    :return: 对应的输出值或输出数组
    """
    x_array = np.asarray(x)
    y = 2 * x_array ** 2 + 3 * x_array + 1

    if np.isscalar(x):
        return float(y)
    return y


def predict_crop_yield(
        soil_fertility: float,
        irrigation_weekly: float,
        avg_temperature: float,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.1
) -> float:
    """
    在农业科研领域，准确预测作物产量对于制定种植策略、优化资源配置具有重要意义。
    本模型基于关键环境与土壤因素，用于估算单位面积上的作物产量。

    模型公式：Y = a * F + b * I - c * T^2

    :param soil_fertility: 土壤肥力指数
    :param irrigation_weekly: 每周灌溉量（mm/week）
    :param avg_temperature: 平均气温（℃）
    :param a: 土壤肥力对产量的贡献系数，默认1.0
    :param b: 灌溉量对产量的贡献系数，默认1.0
    :param c: 温度对产量的抑制系数，默认0.1
    :return: 单位面积作物产量预测值（kg/ha）
    """
    return a * soil_fertility + b * irrigation_weekly - c * avg_temperature ** 2


def first_order_difference_equation(
        y_previous: float,
        x1_t: float,
        x2_t: float,
        x3_t: float,
        x4_t: float,
        x5_t: float,
        a: float = 0.5,
        b: float = 0.2,
        c: float = 0.1,
        d: float = 0.15,
        e: float = 0.05
) -> float:
    """
    本模型构建了一个一阶线性差分方程，用于描述某一系统状态随时间演化的动态行为。
    该系统具有记忆特性，当前状态不仅依赖于当前时刻的输入变量，也受到前一时刻系统状态的影响。

    模型公式：y_t = a * y_{t-1} + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)

    :param y_previous: 前一时刻的系统状态
    :param x1_t: 当前时刻的第一个输入变量
    :param x2_t: 当前时刻的第二个输入变量
    :param x3_t: 当前时刻的第三个输入变量
    :param x4_t: 当前时刻的第四个输入变量
    :param x5_t: 当前时刻的第五个输入变量
    :param a: 历史状态反馈系数，默认0.5
    :param b: 输入变量x1的系数，默认0.2
    :param c: 输入变量x2的系数，默认0.1
    :param d: 输入变量x3的系数，默认0.15
    :param e: 输入变量差值系数，默认0.05
    :return: 当前时刻的系统状态输出y_t
    """
    return a * y_previous + b * x1_t - c * x2_t + d * x3_t + e * (x4_t - x5_t)


def predict_moisture_evaporation(
        initial_moisture: float,
        evaporation_rate: float,
        drying_time: float
) -> dict:
    """
    在食品加工与制造过程中，干燥是一个关键的工艺环节。
    本模型用于预测食品在干燥过程中水分含量的变化和累计蒸发量。

    模型公式：
    1. 水分含量：M(t) = M0 * exp(-k*t)
    2. 累计蒸发量：Evaporated(T) = M0 * (T + (exp(-k*T) - 1)/k)

    :param initial_moisture: 初始水分含量M0
    :param evaporation_rate: 水分蒸发速率常数k
    :param drying_time: 干燥时间T
    :return: 包含水分含量、累计蒸发量和剩余水分的字典
    """
    # 计算最终水分含量
    final_moisture = initial_moisture * math.exp(-evaporation_rate * drying_time)

    # 计算累计蒸发量
    if evaporation_rate != 0:
        evaporated = initial_moisture * (
                    drying_time + (math.exp(-evaporation_rate * drying_time) - 1) / evaporation_rate)
    else:
        evaporated = 0

    # 计算剩余水分
    remaining_moisture = initial_moisture - evaporated if evaporated < initial_moisture else 0

    return {
        "final_moisture_content": final_moisture,
        "total_evaporated": evaporated,
        "remaining_moisture": remaining_moisture,
        "evaporation_percentage": (evaporated / initial_moisture * 100) if initial_moisture > 0 else 0
    }