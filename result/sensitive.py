import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 全局设置（解决中文显示、负号显示问题） --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 适配Windows/其他系统中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['figure.figsize'] = (10, 6)  # 图表大小，可调整
plt.rcParams['font.size'] = 10  # 全局字体大小

# -------------------------- 2. 读取数据并预处理（wifi_mod数值化、排除sample_index） --------------------------
df = pd.read_csv(r"data\battery_simplified_processed_data.csv")

# wifi_mod数值映射（wifi=1、4g=2、5g=4，严格贴合功耗比例），处理异常值
wifi_mod_map = {'wifi': 1, '4g': 2, '5g': 4}
df['wifi_mod'] = df['wifi_mod'].where(df['wifi_mod'].isin(wifi_mod_map.keys()), np.nan)  # 过滤异常值
df['wifi_mod_num'] = df['wifi_mod'].map(wifi_mod_map)  # 数值化
df = df.drop('wifi_mod', axis=1)  # 删除原字符串列

# 筛选计算列：排除第一列sample_index，删除空值行（保证计算有效性）
cols_calc = df.columns[1:].tolist()
df_calc = df[cols_calc].dropna()

# -------------------------- 3. 计算相关性（按绝对值降序，方便可视化对比） --------------------------
# 方法1：皮尔逊相关（适用于线性关系，默认）
corr_pearson = df_calc.corr()['power_consumption_mw'].drop('power_consumption_mw')
corr_pearson_sorted = corr_pearson.sort_values(key=abs)  # 升序排，可视化时从上到下由强到弱

# 方法2：斯皮尔曼相关（适用于非线性/非正态分布，推荐备用）
corr_spearman = df_calc.corr(method='spearman')['power_consumption_mw'].drop('power_consumption_mw')
corr_spearman_sorted = corr_spearman.sort_values(key=abs)

# -------------------------- 4. 相关性可视化（横向柱状图，最适合对比系数） --------------------------
def plot_corr(corr_series, title):
    """
    相关性可视化函数：生成横向柱状图，带正负色区分、数值标注、参考线
    :param corr_series: 排序后的相关性序列
    :param title: 图表标题
    """
    # 定义颜色：正相关蓝色，负相关橙色
    colors = ['#1f77b4' if x > 0 else '#ff7f0e' for x in corr_series.values]
    
    # 绘制横向柱状图
    bars = plt.barh(corr_series.index, corr_series.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    # 添加数值标注（保留3位小数，显示在柱子右侧）
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01 if width > 0 else width - 0.01,  # 标注位置：正负值分开
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
    # 添加辅助线：0值参考线（区分正负相关）
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    
    # 设置图表标题、坐标轴标签
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('相关系数', fontsize=12, labelpad=10)
    plt.ylabel('特征参数', fontsize=12, labelpad=10)
    
    # 调整x轴范围，避免标注超出图表
    x_max = max(abs(corr_series.values)) * 1.2
    plt.xlim(-x_max, x_max)
    
    # 网格线：轻量纵向网格，方便看系数大小
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)  # 隐藏上、右边框，更简洁
    
    # 紧凑布局，避免标签重叠
    plt.tight_layout()
    # 显示图表
    plt.show()

# 绘制皮尔逊相关性图（核心推荐，若数据非正态可替换为corr_spearman_sorted）
plot_corr(corr_pearson_sorted, '功耗(power_consumption_mw)与各参数的皮尔逊相关性')

# 如需绘制斯皮尔曼相关性图，取消注释下方代码
# plot_corr(corr_spearman_sorted, '功耗(power_consumption_mw)与各参数的斯皮尔曼相关性')

# -------------------------- 可选：保存相关性结果和图表 --------------------------
# 保存相关性结果到CSV
corr_pearson_sorted.round(3).to_csv('功耗与各参数相关性结果.csv', header=['皮尔逊相关系数'], encoding='utf-8-sig')
# 保存图表（在plt.show()前执行，格式可选png/pdf，dpi越高越清晰）
# plt.savefig('功耗相关性分析图.png', dpi=300, bbox_inches='tight')