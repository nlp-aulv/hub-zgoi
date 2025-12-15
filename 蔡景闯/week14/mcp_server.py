from fastmcp import FastMCP
import sympy as sp
from typing import Literal

# 创建MCP服务器实例
mcp = FastMCP("MCP Server")

@mcp.tool
def DO(a:float, b:float, c:float, d:float, t:float):
    '''
    溶解氧浓度随时间变化的规律

    :param a: 初始溶解氧释放量，反映系统初始状态下的氧含量
    :param b: 解氧的衰减系数，刻画其随时间自然下降的速率
    :param c: 环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度
    :param d: 环境扰动的频率，反映扰动周期的快慢
    :param t: 时间

    :return: 溶解氧浓度
    '''
    expr = a * sp.exp(-b * t) + c * sp.sin(d * t)
    return expr

@mcp.tool
def orders_predict(ad_spend:float, discount_rate:float, prev_orders:float):
    '''
    当日订单数量预测

    :param ad_spend: 广告支出
    :param discount_rate: 当日折扣力度
    :param prev_orders: 前一日订单数量

    :return: 当日订单数量
    '''
    expr = 0.05 * ad_spend + 100 * discount_rate + 0.7 * prev_orders
    return expr

@mcp.tool
def influence_between_variables(x:float, y:float,):
    '''
    模拟两个输入变量 x 和 y 对某一目标输出的综合影响，其中包含了周期性变化与线性交互的成分。
    该建模方法适用于描述如环境因素对系统响应的影响、多因子耦合作用下的信号响应机制等场景。

    :param x: 输入变量 x
    :param y: 输入变量 y

    :return: 变量作用程度
    '''
    expr = 2.5 * sp.sin(x) + 1.8 * sp.cos(y) + 0.3 * x * y
    return expr

@mcp.tool
def crop_yield_predict(F:float, I:float, T:float):
    '''
    作物产量预测

    :param F: 土壤肥力指数
    :param I: 每周灌溉量
    :param T: 平均气温

    :return: 作物产量
    '''
    a,b,c = sp.symbols('a b c')
    expr = a * F + b * I - c * T**2
    return expr

@mcp.tool
def food_moisture_loss_predict(M0:float, t:float):
    '''
    预测在指定干燥时间内食品的总水分损失量

    :param M0: 初始水分含量
    :param t: 干燥时间

    :return: 水分损失量
    '''
    k = sp.symbols('k')
    expr = M0 * (t + (sp.exp(-k * t) - 1) / k )
    return expr

@mcp.tool
def learning_analysis(x1:int, x2:float, x3:float, x4:Literal[1,2,3,4,5]):
    '''
    通过量化方式反映学生的学习成果，并模拟其在学习过程中的非线性增长趋势。
    该模型可用于学生表现预测、教学反馈分析以及个性化学习路径优化等场景，为教育决策提
    供数据支持。

    :param x1: 学习时间（小时）
    :param x2: 出勤率（百分比）
    :param x3: 平时测验成绩（百分比）
    :param x4: 课堂参与度（1~5）

    :return: 学习成果
    '''
    w1,w2,w3,w4,a,b = sp.symbols('w1 w2 w3 w4 a b')
    expr = 100 / (1 + sp.exp(-a * (w1 * x1 + w2 * x2 + w3 * x3 + w4 * x4 - b)))
    return expr

@mcp.tool
def behavior_predict(x1_t:float, x2_t:float, x3_t:float,y_t1:float, y_t2:float):
    '''
    本建模任务构建了一个包含三个输入变量的差分方程模型，用于模拟并预测系统的下一状态。
    该模型结构简洁，便于分析系统内部变量之间的相互作用，并可用于多变量输入场景下的动态行为预测。

    :param x1_t: 当前时刻的第一个输入变量 
    :param x2_t: 当前时刻的第二个输入变量
    :param x3_t: 当前时刻的第三个输入变量
    :param y_t1: 前一个时间步的输出值
    :param y_t2: 前二个时间步的输出值

    :return: 当前时刻的输出值
    '''
    a,b,c,d = sp.symbols('a b c d')
    expr = a * x1_t + b * y_t1 + c * y_t2 + d * x2_t * x3_t
    return expr

@mcp.tool
def behavior_predict_2(x:float) -> float:
    '''
    该模型基于一个预定义的数学关系，确保在相同输入条件下始终产生一致的输出结果。
    此类模型广泛应用于工程计算、经济预测以及自然科学领域，适用于那些具有明确因果关系的场景

    :param x: 输入变量

    :return: 输出变量
    '''
    expr = 2 * x**2 + 3 * x + 1
    return expr

@mcp.tool
def mid_span_deflection_of_simple_supported_beam(Q:float, L:float, E:float, I:float) -> float:
    '''
    计算简支梁在均布荷载作用下的中点挠度.
    该公式适用于线弹性材料、小变形条件下的简支梁结构，能够有效反映梁在静力荷载下的变形特性。

    :param Q: 均布荷载
    :param L: 梁的长度
    :param E: 材料弹性模量
    :param I: 梁的惯性矩

    :return: 中点挠度
    '''
    expr = (5* Q * L**4) / (384 * E * I)
    return expr

@mcp.tool
def house_value(area:float, floor:float, age:float) -> float:
    '''
    用于估算房产的市场价值，有助于辅助定价、投资决策和市场分析.
    该公式综合考虑了基础单价、面积大小、楼层系数与房龄折旧系数，以量化方式反映各因素对房产市场价值的影响。

    :param area: 房屋面积
    :param floor: 楼层高度
    :param age: 房屋年龄

    :return: 房产价值
    '''
    expr = 10000 * area * (1 + 0.02 * floor) * (1 - 0.015 * age)
    return expr

# 启动服务器 
if __name__ == "__main__":
    # 8900端口运行
    mcp.run(transport='sse', port=8900)
