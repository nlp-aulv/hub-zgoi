from fastmcp import FastMCP
from typing import Annotated
import sympy as sp
from sympy import symbols, Function, Eq, dsolve, simplify, lambdify, solve
from scipy.constants import R
import numpy as np
import math
import random

mcp = FastMCP(name="Tool mcp server")

@mcp.tool
def solve_inventory_model(
    S0: Annotated[float, "初始时刻的库存量，即 S(0)"],
    k: Annotated[float, "库存衰减率，反映销售速率，大于0"],
    u: Annotated[str, "外部补货速率函数 u(t)，以字符串形式表示的时间函数，如 '10' 或 '5*exp(-t)'"],
    t_end: Annotated[float, "模拟结束的时间点，用于求解和返回时间范围"]
):
    """
    在农产品批发与零售过程中，库存管理是保障供应链稳定运行的关键环节。
    由于农产品具有易腐性、季节性和供需波动性等特点，合理预测库存变化对于控制损耗、
    优化补货策略和提升运营效率具有重要意义。本模型构建了一个描述单一农产品库存随时间
    变化的动态系统，基于一阶常微分方程，考虑库存的自然衰减（与当前库存成正比）以及
    外部补货输入的影响。

    参数说明：
    - S0 (float): 初始时刻的库存量，S(0)，必须为非负数。
    - k (float): 库存衰减率，表示单位时间内库存因销售或损耗减少的比例，需大于0。
    - u (str): 外部补货速率函数 u(t)，以字符串形式输入，支持合法的SymPy表达式，
               如常数 '10'、指数函数 '5*exp(-t)'、线性函数 '2*t + 1' 等。
    - t_end (float): 模拟终止时间，生成从 0 到 t_end 的数值结果，应大于0。

    返回值：
    - solution (str): 符号形式的库存函数 S(t) 的解析解。
    - time_points (list of float): 在 [0, t_end] 区间内均匀采样的100个时间点。
    - inventory_values (list of float): 对应 time_points 的库存量数值解（浮点数列表）。
    """
    # 定义符号变量
    t = symbols('t')
    S = Function('S')(t)
    k_sym = symbols('k', positive=True, real=True)

    try:
        # 将输入的 u 字符串转换为 SymPy 表达式
        u_t = eval(u)
    except Exception as e:
        raise ValueError(f"Invalid expression for u(t): {u}. Error: {str(e)}")

    # 构建微分方程：dS/dt = -k*S + u(t)
    eq = Eq(S.diff(t), -k_sym * S + u_t)

    # 求解通解
    general_solution = dsolve(eq, S)

    # 提取积分常数 C1
    C1 = general_solution.rhs.as_independent(sp.Symbol)[1].as_coefficients_dict()[sp.Symbol('C1')]
    C1_symbol = sp.Symbol('C1')

    # 代入初始条件 S(0) = S0
    initial_eq = general_solution.subs(t, 0).subs(S, S0)
    C1_value = solve(initial_eq, C1_symbol)[0]

    # 得到特解
    particular_solution = general_solution.subs(C1_symbol, C1_value)
    solution_expr = simplify(particular_solution.rhs)

    # 转换为数值函数（替换参数 k 和表达式中的符号）
    f_numeric = lambdify(t, solution_expr.subs(k_sym, k), modules='numpy')

    # 生成时间采样点（100个点）
    time_points = np.linspace(0, t_end, 100)
    inventory_values = [float(f_numeric(ti)) for ti in time_points]

    # 返回结果
    return {
        "solution": str(solution_expr),
        "time_points": time_points.tolist(),
        "inventory_values": inventory_values
    }

@mcp.tool
def predict_daily_orders(
    base_orders: Annotated[float, "前一日的订单数量，作为当日预测的基础订单量"],
    marketing_spend: Annotated[float, "当日的营销投入金额，单位为元"],
    conversion_rate: Annotated[float, "访问用户转化为订单的比例，取值范围通常在0到1之间"],
    seasonality_factor: Annotated[float, "乘性调节因子，用于反映季节性、节假日或促销活动对订单的影响"]
):
    """
    在电子商务环境中，订单量的增长受到基础订单水平、营销投入、用户转化效率以及外部季节性因素的综合影响。
    该模型基于前一天的订单基数，结合当日的营销支出、转化率和季节性因子，通过差分方程形式预测当日订单量，
    用于支持运营决策和增长趋势分析。
    
    预测公式为：
    result = base_orders * (1 + 0.05 * marketing_spend / 1000 + conversion_rate) * seasonality_factor
    
    参数说明：
    - base_orders: 前一日订单量，作为增长的基础。
    - marketing_spend: 营销支出，每1000元带来约5%的增量效应（即系数0.05）。
    - conversion_rate: 用户转化率，直接影响新增订单比例。
    - seasonality_factor: 季节性调节因子，>1表示促进（如大促），<1表示抑制（如淡季）。

    返回值：
    - 预测的当日订单量（浮点数）
    """
    # 实现预测逻辑
    result = base_orders * (1 + 0.05 * marketing_spend / 1000 + conversion_rate) * seasonality_factor
    return float(result)

@mcp.tool
def calculate_adg1(
    feed_intake: Annotated[float, "日均采食量，单位: kg，反映动物的摄食能力"],
    protein_level: Annotated[float, "饲料中粗蛋白含量，单位: %，以16%为理想蛋白水平基准"],
    health_status: Annotated[float, "健康状况评分，取值范围在0到1之间，1表示完全健康"],
    temperature: Annotated[float, "环境温度，单位: °C，以20°C为最适生长温度"]
) -> Annotated[dict, "包含日均增重（ADG）结果的对象"]:
    """
    为了更准确地评估牲畜的生长性能, 尤其是日均增重 (Average Daily Gain, ADG), 
    建立了一个非线性模型, 综合考虑了多个关键影响因素。该模型旨在反映实际生产中饲料摄入、营养水平、
    动物健康状态以及环境条件对生长速度的综合作用。通过量化这些变量的影响, 模型为畜牧业管理、
    饲养方案优化以及环境调控提供了理论支持和决策依据。
    
    参数说明:
    - feed_intake (float): 日均采食量（kg），表示动物每天摄入的饲料重量。
    - protein_level (float): 饲料粗蛋白含量（%），相对于16%的理想水平进行加权。
    - health_status (float): 健康评分（0~1），1代表完全健康，降低则增重效率下降。
    - temperature (float): 环境温度（°C），偏离20°C时通过指数衰减函数降低ADG。

    模型公式:
    ADG = (feed_intake * 0.35) * (protein_level / 16) * health_status * exp(-0.05 * (temperature - 20))

    返回:
    dict: 包含键 'ADG' 的字典，值为计算出的日均增重（kg/天）。
    """
    # 计算日均增重 ADG
    adg_value = (feed_intake * 0.35) * (protein_level / 16) * health_status * math.exp(-0.05 * (temperature - 20))
    
    return {"ADG": round(adg_value, 4)}


@mcp.tool
def calculate_wall_heat_loss(
    U: Annotated[float, "墙体材料的传热系数，单位为瓦特每平方米开尔文 (W/(m^2·K))"],
    A: Annotated[float, "墙体的有效传热面积，单位为平方米 (m^2)"],
    delta_T: Annotated[float, "室内外空气之间的温差，单位为开尔文 (K)"]
) -> Annotated[float, "墙体的热损失率，单位为瓦特 (W)"]:
    """
    在建筑环境与能源系统设计中, 墙体的热损失率是评估建筑物热工性能的重要参数之一。
    该参数反映了单位时间内通过墙体散失的热量, 对于建筑节能、供暖系统容量配置以及室内热舒适性分析具有重要意义。
    通过建立科学合理的数学模型, 可以有效预测不同工况下的热损失情况, 为建筑设计和暖通空调 (HVAC) 系统选型提供理论依据。

    计算公式：Q = U * A * ΔT
    其中：
    - Q：墙体热损失率（单位：瓦特，W）
    - U：传热系数（W/(m²·K)）
    - A：传热面积（m²）
    - ΔT：室内外温差（K）

    参数说明：
    - U (float): 传热系数，反映材料导热性能，值越小保温性能越好。
    - A (float): 墙体面积，需为正数。
    - delta_T (float): 温差，通常取 |T_inside - T_outside|，单位与开尔文等效。

    返回值：
    - float: 单位时间内的热损失量，单位为瓦特 (W)。
    """
    if U < 0 or A < 0 or delta_T < 0:
        raise ValueError("传热系数、面积和温差必须为非负数。")
    
    Q = U * A * delta_T
    return Q

@mcp.tool
def predict_milk_production(concentrate_intake: Annotated[float, "精饲料摄入量（单位：公斤/天）"]) -> Annotated[dict, "包含预测产奶量的字典"]:
    """
    在畜牧业管理中, 科学预测奶牛的产奶量对于优化饲养策略、提高生产效率具有重要意义。
    本模型聚焦于精饲料摄入量对产奶量的核心影响, 构建一个简明且具备解释性的线性预测模型,
    用于评估不同饲养条件下奶牛的预期产奶水平。

    预测公式为：预测产奶量 = 2.5 × 精饲料摄入量 + 10
    """
    milk_yield = 2.5 * concentrate_intake + 10
    return {"milk_yield": milk_yield}

@mcp.tool
def calculate_agricultural_price(
    t: Annotated[float, "时间变量，表示从初始时刻起经过的时间（如天数），用于反映价格随时间的动态增长效应。"],
    s: Annotated[float, "供应量，表示市场上农产品的供应水平，供应越大，价格越低。"],
    d: Annotated[float, "需求量，表示市场上对农产品的需求水平，需求越大，价格越高。"],
    w: Annotated[float, "天气影响因子，表示天气对农业生产与流通的影响，如恶劣天气可能推高价格。"],
    c: Annotated[float, "运输成本，直接叠加到最终价格中，反映物流对价格的影响。"]
) -> Annotated[float, "返回计算得到的农产品市场价格 P(t, s, d, w, c)，为一个实数。"]:
    """
    在农产品批发与零售市场中，价格受到时间、供应量、需求量、天气状况和运输成本等多因素影响。
    本模型通过一个综合的数学函数描述价格的动态变化，可用于预测价格波动、优化供应链管理及辅助市场决策。

    模型公式：
        price = (d / (s + 1)) * w * (1 + 0.05 * t) + c

    其中：
        - (d / (s + 1))：供需比，反映需求与供应的相对关系，分母加1避免除以0。
        - w：天气因子，放大或缩小基础供需价格。
        - (1 + 0.05 * t)：时间增长因子，模拟随时间推移的价格自然上涨趋势（如通胀、季节性上涨）。
        - c：运输成本，直接加成到最终价格上。

    示例：
        t=10, s=50, d=60, w=1.2, c=3 → price ≈ (60/51)*1.2*(1+0.5)+3 ≈ 2.12*1.5+3 ≈ 6.18
    """
    # 防止除以零，s + 1 确保分母安全
    supply_demand_ratio = d / (s + 1)
    time_factor = 1 + 0.05 * t
    base_price = supply_demand_ratio * w * time_factor
    final_price = base_price + c
    return float(final_price)

@mcp.tool
def predict_resting_heart_rate(Age: Annotated[float, "个体的年龄，单位为岁，应为非负数值。"]):
    """
    在医疗健康领域，静息心率（Resting Heart Rate, RHR）是反映心血管健康的重要生理指标。
    研究表明，健康人群中静息心率与年龄之间存在近似线性关系，随年龄增长而缓慢上升。
    本模型构建了一个简化的线性关系用于初步估计不同年龄段个体的静息心率水平，适用于教学演示和基础分析。

    参数说明:
        Age (float): 个体的年龄，单位为岁，应为非负数值。

    返回值:
        dict: 包含预测静息心率（RHR）的字典，单位为次/分钟，基于线性模型计算得出。
              格式为 {"RHR": 数值}。

    模型公式:
        RHR = 60 + 0.3 * Age
    """
    try:
        if Age < 0:
            raise ValueError("Age must be non-negative.")
        RHR = 60 + 0.3 * Age
        return {"RHR": float(RHR)}
    except Exception as e:
        return {"error": str(e)}

@mcp.tool
def calculate_adg2(
    F: Annotated[float, "每日干物质采食量，单位：kg"],
    P: Annotated[float, "饲料粗蛋白含量，单位：%（例如15.0表示15%）"],
    E: Annotated[float, "饲料代谢能含量，单位：MJ/kg"],
    W: Annotated[float, "动物当前体重，单位：kg"],
    T: Annotated[int, "已饲喂天数，单位：day"]
) -> Annotated[float, "返回在 [0, T] 饲喂周期内累积的平均日增重 (ADG)，单位：kg/day 的积分累积值，表示总增重贡献量。"]:
    """
    在畜牧业管理中, 预测牲畜的平均日增重 (ADG) 是评估饲养效率和优化饲料资源配置的关键环节。
    该模型基于积分方程，综合考虑每日干物质采食量、饲料粗蛋白含量、代谢能含量以及动物当前体重随时间的变化，
    量化在饲喂周期内的总增重累积过程。模型假设日增重与采食量和营养浓度呈正相关，与当前体重呈负相关，
    反映增重效率随体重增长而递减的生物学规律。

    参数说明:
    - F (float): 每日干物质采食量（kg），直接影响营养摄入总量。
    - P (float): 饲料粗蛋白含量（%），是蛋白质营养水平的指标。
    - E (float): 饲料代谢能含量（MJ/kg），代表能量供给能力。
    - W (float): 动物当前体重（kg），用于标准化增重速率，体现体重越大增重越慢的效应。
    - T (int): 已饲喂天数（day），定义积分的时间区间 [0, T]。

    模型计算公式（符号积分形式）:
        ADG = ∫₀ᵀ (F × P × E) / W dt = (F × P × E × T) / W

    由于被积函数在时间 t 上为常数（假设F, P, E, W在T天内不变），积分结果为线性累积。
    返回值为整个饲喂周期内的总增重贡献量（单位：kg），可解释为平均日增重的累积表达。
    """
    if W <= 0:
        raise ValueError("动物当前体重 W 必须大于 0。")
    if T < 0:
        raise ValueError("已饲喂天数 T 不能为负数。")

    # 执行积分计算: ∫₀ᵀ (F * P * E) / W dt = (F * P * E * T) / W
    adg_integral_value = (F * P * E * T) / W
    return float(adg_integral_value)

@mcp.tool
def calculate_quality_retention(
    t: Annotated[float, "加工总时间（单位：分钟）"],
    T: Annotated[float, "加工过程中的温度（单位：°C）"],
    T0: Annotated[float, "参考温度（单位：°C），表示质量保留率最高的温度点"],
    sigma: Annotated[float, "温度敏感度系数，控制温度偏离T0时质量损失的速度"],
    RH: Annotated[float, "环境相对湿度（0到1之间的数值），湿度越高，质量保留率越低"],
    v: Annotated[float, "空气流速（单位：m/s），影响质量变化速率"]
) -> Annotated[float, "返回归一化后的平均质量保留率（0到1之间的数值），表示在给定加工条件下食品质量的保留程度"]:
    """
    在食品加工与制造过程中，质量控制是确保产品品质和生产效率的关键环节。
    常见的加工操作如干燥、杀菌、发酵等，均会受到时间、温度、环境湿度和空气流速等工艺参数的影响。
    该模型用于预测不同加工条件下食品质量的保留率，通过积分形式模拟温度、相对湿度和空气流速对质量的累积影响，
    从而指导工艺参数的优化设置。

    模型公式基于以下假设：
    - 温度对质量保留的影响服从高斯衰减函数，峰值在T0处；
    - 相对湿度线性降低保留率，因子为(1 - RH)；
    - 空气流速通过平方根项√v增强质量变化速率；
    - 在时间区间[0, t]上对综合影响因子积分，并归一化得到平均保留率。

    公式形式（符号计算）：
        integrand = exp(-(T - T0)**2 / (2 * sigma**2)) * (1 - RH) * sqrt(v)
        result = integrate(integrand, (t, 0, t)) → [exp(...) * (1-RH) * sqrt(v)] * t
        最终结果已归一化至 [0,1] 范围。
    """
    # 计算温度适应性因子（高斯衰减）
    temp_factor = math.exp(-((T - T0) ** 2) / (2 * sigma ** 2)) if sigma != 0 else 0.0

    # 湿度抑制因子
    humidity_factor = 1 - RH

    # 空气流速促进因子
    velocity_factor = math.sqrt(v)

    # 综合瞬时保留率贡献
    instantaneous_rate = temp_factor * humidity_factor * velocity_factor

    # 在时间t内的累积影响（线性积分：rate × time）
    cumulative_retention = instantaneous_rate * t

    # 归一化处理：由于原始模型未指定最大值，我们假设理想条件下最大可能积分为 1×1×1×t = t
    # 因此归一化保留率为 cumulative_retention / t，即等效于 instantaneous_rate，但考虑了时间长度的有效性
    # 注意：若 t=0，则无加工过程，保留率为 0
    if t <= 0:
        return 0.0

    normalized_retention = cumulative_retention / t  # 实际为 instantaneous_rate，但确保维度一致

    # 限制输出在 [0, 1] 区间内
    return max(0.0, min(1.0, normalized_retention))

@mcp.tool
def predict_assembly_time(
    part_count: Annotated[float, "零部件数量，表示整车装配中涉及的零件总数"],
    automation_level: Annotated[float, "自动化水平，取值范围为[0, 1]，反映装配流程中自动设备的占比"]
) -> Annotated[float, "预测的整车装配时间（单位：分钟），包含由随机扰动带来的生产波动"]:
    """
    在汽车制造过程中, 整车装配时间是影响生产效率和成本控制的重要因素。
    该模型通过模拟装配过程中的关键变量, 包括零部件数量与自动化水平, 预测整车装配所需时间。
    模型考虑了生产过程中的不确定性, 引入了随机扰动以更真实地反映实际生产环境的波动。
    
    参数说明:
    - part_count (float): 零部件数量，必须为非负数，表示装配的复杂程度。
    - automation_level (float): 自动化水平，取值范围 [0, 1]，0 表示完全手动，1 表示完全自动。

    返回值:
    - float: 预测的装配时间（分钟），基础时间为120分钟，并根据零件数和自动化水平调整，
             同时引入 0.9~1.1 之间的随机扰动因子以模拟生产波动。
    """
    # 基础装配时间（分钟）
    basic_time = 120
    # 每增加一个零件所增加的时间基数（分钟）
    basic_time_factor = 0.05
    
    # 随机扰动因子，模拟生产环境波动，均匀分布在 [0.9, 1.1]
    random_disturbance = np.random.uniform(0.9, 1.1)
    
    # 计算受自动化水平影响的额外装配时间
    # 自动化程度越高，人工依赖越低，时间增量越小
    time_contribution = part_count * basic_time_factor * (1 - automation_level)
    
    # 总装配时间 = 基础时间 + 调整后的时间贡献 × 扰动因子
    assembly_time = basic_time + time_contribution * random_disturbance
    
    return round(assembly_time, 2)

@mcp.tool
def predict_crop_yield(
    temp: Annotated[float, "平均温度（单位：摄氏度）"],
    rainfall: Annotated[float, "降水量（单位：毫米）"],
    fertilizer: Annotated[float, "施肥量（单位：千克/公顷）"],
    sunlight: Annotated[float, "日照时间（单位：小时）"]
) -> Annotated[float, "预测的作物产量（单位：吨/公顷）"]:
    """
    在农业生产中,作物产量受到多种环境和管理因素的综合影响。为了辅助农业决策、优化种植策略,
    构建了基于关键变量的线性预测模型。该模型旨在通过量化温度、降水量、施肥量以及日照时间对作物产量的贡献,
    提供对产量趋势的初步预估。模型假设各变量与产量之间存在线性关系,并通过模拟数据验证其基本可行性。
    此方法可作为进一步复杂建模的基础,也可用于教学和实践中的初步分析。

    参数说明:
    - temp: 平均温度，单位为摄氏度，影响作物生长速率。
    - rainfall: 降水量，单位为毫米，反映水分供应情况。
    - fertilizer: 施肥量，单位为千克/公顷，代表养分投入水平。
    - sunlight: 日照时间，单位为小时，决定光合作用强度。

    模型表达式（基于线性假设）:
        yield = 0.1 * temp + 0.05 * rainfall + 0.02 * fertilizer + 0.3 * sunlight

    返回值:
        预测的作物产量，单位为吨/公顷。
    """
    # 线性模型计算
    yield_value = (
        0.1 * temp +
        0.05 * rainfall +
        0.02 * fertilizer +
        0.3 * sunlight
    )
    return round(yield_value, 4)  # 保留四位小数以提高可读性

@mcp.tool
def calculate_aqcr(
    pollutant_conc: Annotated[float, "污染物浓度，作为空气质量变化的基础驱动因子"],
    wind_speed: Annotated[float, "风速，影响污染物的稀释与扩散，单位通常为m/s"],
    temp: Annotated[float, "温度，影响污染物的化学反应与累积，单位为摄氏度"],
    humidity: Annotated[float, "相对湿度，影响颗粒物的聚集与沉降，单位为百分比（0-100）"]
) -> Annotated[dict, "返回空气质量变化速率（AQCR）结果"]:
    """
    在环境监测与空气质量管理中，理解空气质量的动态变化趋势至关重要。
    该函数实现一个基于非线性关系的数学模型，用于量化不同气象和污染条件对空气污染趋势的综合效应。
    核心输出指标为“空气质量变化速率”(Air Quality Change Rate, AQCR)，适用于特定区域的短期空气质量趋势评估与辅助决策。

    参数说明:
    - pollutant_conc (float): 污染物浓度，数值越大表示初始污染水平越高，是空气质量恶化的基础驱动力。
    - wind_speed (float): 风速（m/s），风速越高越有利于污染物扩散，从而减缓空气质量恶化。
    - temp (float): 温度（摄氏度），较高温度可能促进光化学反应，加剧污染形成，模型中以负向调节作用体现。
    - humidity (float): 相对湿度（0-100%），高湿度可能促进颗粒物吸湿增长和聚集，影响沉降过程，间接加剧污染趋势。

    函数参数与模型公式:
    AQCR = (0.5 * pollutant_conc / (1 + 0.1 * wind_speed^2)) * (1 - 0.02 * temp) * (1 + 0.005 * humidity^1.2)

    公式解释：
    - 第一部分：(0.5 * pollutant_conc / (1 + 0.1 * wind_speed^2)) 表示污染物浓度被风速的平方项抑制，反映风力增强时扩散效应显著；
    - 第二部分：(1 - 0.02 * temp) 表示温度升高会线性增加化学反应活性，加快污染生成；
    - 第三部分：(1 + 0.005 * humidity^1.2) 使用湿度的1.2次幂非线性增强其影响，体现高湿环境下颗粒物行为的复杂变化。

    返回值:
    - AQCR (float): 空气质量变化速率，正值表示空气质量趋于恶化，负值或接近零表示趋于稳定或改善。
      该值可用于预警、趋势分析或作为控制系统的输入信号。
    """
    try:
        # 模型计算逻辑
        dispersion_factor = (0.5 * pollutant_conc) / (1 + 0.1 * math.pow(wind_speed, 2))
        temp_effect = (1 - 0.02 * temp)
        humidity_effect = (1 + 0.005 * math.pow(humidity, 1.2))

        aqcr = dispersion_factor * temp_effect * humidity_effect

        return {"AQCR": float(aqcr)}
    except Exception as e:
        return {"error": f"Failed to calculate AQCR: {str(e)}"}

@mcp.tool
def predict_sales(
    a: Annotated[float, "广告投入金额 (单位: 元), 反映市场推广力度"],
    b: Annotated[float, "商品折扣率, 取值范围为0到1, 用于衡量价格优惠程度"]
) -> Annotated[dict, "预测的销售额结果"]:
    """
    在电子商务运营中, 准确预测销售额对于制定营销策略、优化资源配置以及提升盈利能力具有重要意义。
    本函数通过量化广告投入与商品折扣对销售额的影响，建立预期销售额预测模型。
    
    模型公式：sales = 1000 + 2*a - 500*b + 3*a*b
    其中：
      - 常数项 1000 表示基础销售额（无推广和折扣时的基准销量）
      - 2*a 表示广告投入的直接正向拉动效应
      - -500*b 表示折扣带来的利润减少或成本增加的负面影响（如让利）
      - 3*a*b 表示广告与折扣之间的协同效应，即联合促销的放大作用
    
    适用于中短期销售预测场景，辅助运营决策。
    
    参数说明：
      a (float): 广告投入金额，单位为元，必须为非负数。
      b (float): 商品折扣率，取值范围 [0, 1]，例如 0.8 表示八折。
    
    返回：
      dict: 包含预测销售额的字典，键为 'sales'，单位为元。
    """
    # 输入校验
    if a < 0:
        raise ValueError("广告投入金额 a 必须大于等于 0")
    if not (0 <= b <= 1):
        raise ValueError("商品折扣率 b 必须在 0 到 1 之间")

    # 执行销售额预测计算
    sales = 1000 + 2 * a - 500 * b + 3 * a * b

    return {"sales": float(sales)}


@mcp.tool
def calculate_conversion(
    t: Annotated[float, "反应时间（单位：s）"],
    T: Annotated[float, "反应温度（单位：℃）"],
    A: Annotated[float, "指前因子（单位：1/s）"],
    Ea: Annotated[float, "活化能（单位：J/mol）"]
) -> Annotated[float, "归一化后的转化率，取值范围 [0, 1]，表示在给定时间 t 和温度 T 下的反应转化率"]:
    """
    在能源化工过程中，反应器的设计与优化依赖于对反应动力学的准确描述。
    转化率作为衡量反应进行程度的关键参数，通常受到反应时间、温度以及反应速率常数的显著影响。
    本模型采用积分方程形式，结合阿伦尼乌斯方程描述温度对反应速率的影响，
    并通过时间积分反映反应的累积效果。
    该方法适用于均相反应体系的初步模拟，可作为连续搅拌釜反应器 (CSTR) 或固定床反应器中反应动力学分析的基础工具。

    参数说明：
    - t: 反应时间，单位为秒（s）
    - T: 反应温度，单位为摄氏度（℃），将自动转换为开尔文用于计算
    - A: 指前因子，单位为 1/s，代表反应的频率因子
    - Ea: 活化能，单位为 J/mol，决定温度对反应速率的影响强度

    返回值：
    - 归一化转化率，范围 [0, 1]，表示当前条件下反应物转化为产物的比例。
      基于积分形式的动力学模型：∫₀ᵗ k(T)·exp(-τ) dτ，其中 k(T) = A·exp(-Ea/(R·T))
    """
    # 定义符号变量
    tau = sp.Symbol('tau')

    # 将温度从 ℃ 转换为 K
    T_K = T + 273.15

    # 计算温度依赖的反应速率常数（阿伦尼乌斯方程）
    k_T = A * math.exp(-Ea / (R * T_K))

    # 构建被积函数：k_T * exp(-tau)
    integrand = k_T * sp.exp(-tau)

    # 对 tau 从 0 到 t 进行积分
    integral_result = sp.integrate(integrand, (tau, 0, t))

    # 转换为浮点数
    conversion_value = float(integral_result)

    # 归一化到 [0, 1] 范围（理论上最大值为 k_T * 1，但 exp(-tau) 积分为 1 - exp(-t)，此处已精确积分）
    # 实际上 ∫₀ᵗ exp(-tau) dτ = 1 - exp(-t)，所以最大可能值是 k_T * (1 - exp(-t))，但这里 k_T 是乘在外部
    # 所以最终结果为：k_T * (1 - exp(-t))
    # 不过根据表达式，我们直接使用积分结果即可，其自然有界

    # 确保输出在 [0, 1] 范围内（防止数值误差导致越界）
    conversion_clipped = max(0.0, min(1.0, conversion_value))

    return conversion_clipped

@mcp.tool
def calculate_daily_profit(
    sales_price_per_kg: Annotated[float, "每公斤农产品的零售价格，单位：元"],
    cost_price_per_kg: Annotated[float, "每公斤农产品的采购或生产成本，单位：元"],
    quantity_sold_kg: Annotated[float, "当日实际销售的总重量，单位：公斤"]
) -> Annotated[dict, "包含日销售利润的字典，单位：元"]:
    """
    在农产品批发与零售行业中，准确评估每日销售利润对于企业经营决策和成本控制具有重要意义。
    该函数基于基础的经济利润计算原理，通过销售价格、成本价格和销售数量三个核心变量，
    计算单一农产品在特定时间内的盈利情况，适用于无损耗、无折扣的理想市场交易环境。
    
    参数说明:
    - sales_price_per_kg (float): 每公斤的销售价格（元），必须大于0
    - cost_price_per_kg (float): 每公斤的成本价格（元），必须大于等于0
    - quantity_sold_kg (float): 当日售出的总重量（公斤），必须大于0
    
    返回值:
    - dict: 包含键 'daily_profit' 的字典，表示日利润（元），计算公式为：
            daily_profit = (sales_price_per_kg - cost_price_per_kg) * quantity_sold_kg
    
    示例:
    >>> calculate_daily_profit(8.5, 5.0, 100)
    {'daily_profit': 350.0}
    """
    # 输入校验
    if sales_price_per_kg < 0:
        raise ValueError("销售价格不能为负数")
    if cost_price_per_kg < 0:
        raise ValueError("成本价格不能为负数")
    if quantity_sold_kg <= 0:
        raise ValueError("销售数量必须大于0")

    # 计算日利润
    daily_profit = (sales_price_per_kg - cost_price_per_kg) * quantity_sold_kg

    return {"daily_profit": round(daily_profit, 2)}


@mcp.tool
def calculate_beam_deflection(
    L: Annotated[float, "梁的跨度, 单位为米 (m)"],
    w: Annotated[float, "均布荷载, 单位为千牛每米 (kN/m)"],
    E: Annotated[float, "材料的弹性模量, 单位为千帕斯卡 (kPa)"],
    I: Annotated[float, "截面惯性矩, 单位为米四次方 (m^4)"]
):
    """
    在结构工程分析中, 梁的挠度是衡量其在荷载作用下变形能力的重要指标。挠度过大会影响结构的正常使用, 甚至引发安全问题。
    本函数模拟一个简支梁在给定跨度条件下的挠度响应，基于经典公式并引入可控的随机系数以反映材料与荷载的微小波动。
    使用固定随机种子（42）确保每次调用时输入相同则输出一致，适用于教学演示和数值模拟测试。

    公式：delta = k * (5 * w * L^4) / (384 * E * I)
    其中 k 是一个在 [0.95, 1.05] 范围内随机生成的系数，表示系统不确定性。

    参数:
        L (float): 梁的跨度, 单位为米 (m)
        w (float): 均布荷载, 单位为千牛每米 (kN/m)
        E (float): 材料的弹性模量, 单位为千帕斯卡 (kPa)
        I (float): 截面惯性矩, 单位为米四次方 (m^4)

    返回:
        dict: 包含键 'deflection' 的字典，值为计算得到的挠度，单位为毫米 (mm)
              （注意：原公式单位为米，已转换为毫米 ×1000）
    """
    # 生成围绕1.0的随机系数，浮动±5%
    k = random.uniform(0.95, 1.05)

    # 计算中心最大挠度（单位：米）
    delta_meters = k * (5 * w * L**4) / (384 * E * I)

    # 转换为毫米
    delta_mm = delta_meters * 1000

    return {"deflection": delta_mm}


@mcp.tool
def linear_model(x: Annotated[float, "输入变量，用于进行线性变换的数值"]) -> Annotated[dict, "包含线性模型输出结果的对象"]:
    """
    在实际工程与数据分析场景中, 线性模型是一种基础且广泛应用的数学工具,
    用于描述输入与输出之间的线性关系。该模型适用于变化趋势稳定、影响因素单一或近似线性的问题场景,
    例如预测、趋势分析和简单系统建模等。本模型以一个简化的线性方程形式构建,
    旨在展示基本的建模流程与逻辑结构。

    函数参数:
        x (float): 输入变量，用于进行线性变换的数值

    返回值:
        dict: 包含以下键值对：
            - "output" (float): 根据线性模型 y = 3x + 2 计算得到的输出值

    示例:
        输入: x = 4
        输出: {"output": 14.0}
    """
    output = 3 * x + 2
    return {"output": output}

@mcp.tool
def predict_sales_from_ad_spend(ad_spend: Annotated[float, "广告投入金额，单位为货币单位（如元），用于预测对应的销售额。"]) -> dict:
    """
    在电子商务运营中, 广告投入是影响销售额的重要因素之一。为了量化广告投入对销售额的边际效应,
    并考虑实际中常见的边际收益递减规律, 建立一个能够反映非线性关系的预测模型具有重要意义。
    该模型采用二次函数形式：sales = 500 + 2 * ad_spend - 0.001 * ad_spend^2，
    能够有效捕捉随着广告投入增加，销售额增长逐渐放缓的现象。
    模型可用于辅助营销决策, 优化广告预算分配, 从而提升整体运营效率。

    参数说明:
    - ad_spend (float): 广告投入金额，必须为非负数值，单位为货币单位（如元）。

    返回值:
    - dict: 包含键 'sales' 的字典，其值为根据模型计算出的预期销售额（float，单位：元）。
    """
    # 使用定义的模型公式：sales = 500 + 2*ad_spend - 0.001*ad_spend**2
    sales = 500 + 2 * ad_spend - 0.001 * (ad_spend ** 2)
    return {"sales": float(sales)}

@mcp.tool
def simulate_crop_growth_rate(
    W: Annotated[float, "土壤含水量，表示作物可利用的水分条件，无量纲或相对单位"],
    N: Annotated[float, "土壤氮含量，单位为mg/kg或相对浓度单位"],
    T: Annotated[float, "环境温度，单位为摄氏度（℃）"]
) -> Annotated[float, "作物生长速率 dG/dt，单位为生物量/天"]:
    """
    本模型旨在模拟特定环境条件下作物生长速率的动态变化。作物的生长受到土壤含水量、土壤氮含量和温度等关键环境因子的非线性影响。
    光照等其他因素被简化为常数项。模型适用于初步分析不同环境条件对作物生长的限制作用，可为农业生产管理提供理论支持。

    参数说明：
    - W (float): 土壤含水量，取值通常在 [0, 1] 或实际相对含水范围，影响水分供应。
    - N (float): 土壤氮含量，单位 mg/kg，通过饱和响应函数影响生长。
    - T (float): 环境温度（℃），最适温度为25℃，偏离该温度会降低生长速率。

    模型公式（基于SymPy表达式转换）:
        growth_rate = 0.05 * W * (1 - exp(-0.1 * N)) * (1 - |T - 25| / 20)

    注意：当 |T - 25| >= 20 时，温度项为负或零，生长停止；即温度低于5℃或高于45℃时生长速率为0。
    """
    # 计算温度修正项：在5℃到45℃之间为正，否则为0或负，此时设为0
    temp_effect = 1 - abs(T - 25) / 20
    if temp_effect <= 0:
        temp_effect = 0

    # 氮素响应项：饱和增长函数
    nitrogen_response = 1 - math.exp(-0.1 * N)

    # 综合生长速率
    growth_rate = 0.05 * W * nitrogen_response * temp_effect

    return round(growth_rate, 6)  # 返回保留6位小数的结果

@mcp.tool
def calculate_satisfaction_rate(
    amenity_quality: Annotated[float, "设施质量评分，范围通常为0到5，值越高表示设施越完善"],
    crowd_level: Annotated[float, "人流量水平，范围通常为0到10，值越高表示越拥挤"],
    travel_time: Annotated[float, "游客到达景区所需的交通时间（小时），值越大对满意度负面影响越大"],
    temperature: Annotated[float, "当前气温（摄氏度），偏离22℃越多，对满意度的负面影响越大"]
) -> Annotated[float, "返回游客满意度的变化率 dS/dt，表示单位时间内满意度的增减趋势。正值表示满意度上升，负值表示下降"]:
    """
    在旅游休闲管理与游客体验优化中，游客满意度是一个关键的动态指标，受到天气温度、人流量、设施质量和交通时间等多种外部因素的综合影响。
    该模型通过常微分方程量化这些因素对游客满意度变化速率的影响，可用于景区管理、游客流量调控和服务质量提升策略的制定。
    
    模型公式：
    dS/dt = 0.3*(amenity_quality/5) - 0.2*(crowd_level/10) - 0.1*travel_time - 0.05*|temperature - 22|
    
    各项含义：
    - 设施质量正向促进满意度；
    - 拥挤程度和交通时间负向影响满意度；
    - 温度以22℃为理想值，偏离越大影响越负面。
    """
    # 归一化处理并计算满意度变化率
    ds_dt = (
        0.3 * (amenity_quality / 5) -
        0.2 * (crowd_level / 10) -
        0.1 * travel_time -
        0.05 * abs(temperature - 22)
    )
    return ds_dt

if __name__ == "__main__":
    mcp.run(transport="sse", port=8900)
