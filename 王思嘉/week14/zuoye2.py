from fastmcp import FastMCP
import sympy as sp


# 创建MCP服务器实例
mcp = FastMCP("MCP Server")

@mcp.tool
#1.用于复杂系统分析中模拟两个输入变量 x 和 y 对某一目标输出的综合影响
def fun(x:float, y:float):
    expr = 2.5 * sp.sin(x) + 1.8 * sp.cos(y) + 0.3 * x * y
    return expr

@mcp.tool  
def DO(a:float, b:float, c:float, d:float, t:float):
   
    '''
    2.可用于模拟封闭或半封闭养殖系统中DO浓度随时间演变的过程，为水质调控和管理提供理论支持。其中：
    a为初始溶解氧释放量，反映系统初始状态下的氧含量；
    b 为溶解氧的衰减系数，刻画其随时间自然下降的速率；
    c为表示环境扰动的振幅，体现外部周期性因素（如昼夜变化）对DO浓度的影响强度，
    d为环境扰动的频率，反映扰动周期的快慢；
    t 表示时间；
    '''
    expr = a * sp.exp(-b * t) + c * sp.sin(d * t)
    return expr

@mcp.tool
def orders_predict(ad_spend:float, discount_rate:float, prev_orders:float):
    '''
    3.适用于短期订单趋势和敏感性分析，其中：
    ad_spend 表示广告支出；
    discount_rate 表示折扣率
    prev_orders 表示前一日订单数量
    0。05是广告支出对订单量的敏感系数，0.7是前一日订单对当前订单数量的影响，100是折扣力度的权重因子
    '''
    expr = 0.05 * ad_spend + 100 * discount_rate + 0.7 * prev_orders
    return expr

@mcp.tool
def yield_predict(F:float, l:float, t:float):
    '''
    4.适用于在可控环境条件下对产量进行定量分析和趋势预测。其中：
    $ F $为土壤肥力指数；
    $ I $ 为每周灌溉量（mm/week），
    $ T $ 表示为平均气温（℃）,
    a,b,c为参数，反映作物产量随三个输入变量的组合影响。(设定为常数)
    '''
    a,b,c=1   #这里用的时候记得改成自己的参数
    expr = a * F + b * l - c * t**2
    return expr

@mcp.tool
def moni(m0:float, k:float, t:float):
    '''
    5.适用于可作为干燥过程模拟与优化的基础工具，其中：
    moni->t时刻食品的水分含量；
    m0 表示初始水分含量;
    k 表示水分蒸发速率常数;
    t表示干燥时间
    '''
    expr = m0 * sp.exp(-k * t)
    return expr

@mcp.tool #介于本人差分方程不太好可能这样写有点问题。。。
def y_t(x1:float, x2:float, x3:float, y_t1:float, y_t2:float):
    '''
    6.适用于多变量输入场景下的动态行为预测，其中：
    y_t ：表示第当前时刻的输出值；
    x1,x2,x3表示当前时刻的三个输入变量
    y_t1 ：表示前一时刻的输出值；
    y_t2 ：表示前两时刻的输出值；
    a,b,c,d为参数，反映行为随三个输入变量的组合影响。(设定为常数)
    '''
    a,b,c,d = 1
    expr = a * x1 + b * y_t1 + c * y_t2 + d * x2 * x3
    return expr

@mcp.tool
def score(x1:float, x2:float, x3:float,x4:float):
    '''
    7.适用于学生表现预测、教学反馈分析以及个性化学习路径优化等场景，其中：
    x1 表示学习时长（小时）；
    x2 表示出勤率（百分比）;
    x3 表示平时测验平均分（百分比）
    x4 表示课堂参与度（1~5分），经过线性映射后参与计算
    这里a=0.05 用于控制S型曲线的陡峭程，b=0.3 控制曲线在横轴上的平移位，各w参数可根据需要修改
    '''
    expr = 100/(1+sp.exp(-0.05*(x1+x2*0.01+x3*0.01+x4*0.2-0.3)))
    return expr

@mcp.tool
def y(x:float):
    '''
    8.适用于那些具有明确因果关系的场景,如工程计算、经济预测以及自然科学领域
    '''
    expr = 2 * x**2 + 3 * x + 1
    return expr

@mcp.tool
def influence(con_q:float, channel:float, engage:float,t:float):
    '''
    9.适用于模拟影响力随时间的累积过程，并提供可计算的综合影响力指标，其中：
    con_q 表示内容质量的高低；
    channel 表示传播广度的大小
    engage 表示受众活跃程度的强弱
    t 表示传播周期长短
    '''
    expr = con_q * channel * engage * t
    return expr

@mcp.tool#也是差分方程，估计函数需要改233
def n_t1(nt:float, r:float, k:float):
    '''
    10.适用于对中长期牛群数量趋势进行预测，辅助制定合理的养殖策略与资源分配方案。其中：
    $ Nt $ 表示第 $ t $ 年的牛群数量；
    r 表示年增长率
    $ K $ 为环境承载能力    
    '''
    expr = nt * r / (1 + k * nt)
    return expr

# 启动服务器 
if __name__ == "__main__":
    # 8900端口运行
    mcp.run(transport='sse', port=8900)


