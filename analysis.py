import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  
plt.rcParams['axes.unicode_minus'] = False  
plt.rcParams['figure.figsize'] = (10, 6)  
plt.rcParams['font.size'] = 10  


df = pd.read_csv(r"data\battery_simplified_processed_data.csv")

wifi_mod_map = {'wifi': 1, '4g': 2, '5g': 4}
df['wifi_mod'] = df['wifi_mod'].where(df['wifi_mod'].isin(wifi_mod_map.keys()), np.nan)  
df['wifi_mod_num'] = df['wifi_mod'].map(wifi_mod_map) 
df = df.drop('wifi_mod', axis=1)

cols_calc = df.columns[1:].tolist()
df_calc = df[cols_calc].dropna()

corr_pearson = df_calc.corr()['power_consumption_mw'].drop('power_consumption_mw')
corr_pearson_sorted = corr_pearson.sort_values(key=abs) 

corr_spearman = df_calc.corr(method='spearman')['power_consumption_mw'].drop('power_consumption_mw')
corr_spearman_sorted = corr_spearman.sort_values(key=abs)

def plot_corr(corr_series, title):

    colors = ['#1f77b4' if x > 0 else '#ff7f0e' for x in corr_series.values]
    bars = plt.barh(corr_series.index, corr_series.values, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01 if width > 0 else width - 0.01, 
                 bar.get_y() + bar.get_height()/2,
                 f'{width:.3f}',
                 ha='left' if width > 0 else 'right', va='center', fontsize=9)
    
 
    plt.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('相关系数', fontsize=12, labelpad=10)
    plt.ylabel('特征参数', fontsize=12, labelpad=10)
    
    x_max = max(abs(corr_series.values)) * 1.2
    plt.xlim(-x_max, x_max)
    
    plt.grid(axis='x', linestyle=':', alpha=0.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False) 
    plt.tight_layout()
    plt.show()

plot_corr(corr_pearson_sorted, '功耗(power_consumption_mw)与各参数的皮尔逊相关性')
corr_pearson_sorted.round(3).to_csv('功耗与各参数相关性结果.csv', header=['皮尔逊相关系数'], encoding='utf-8-sig')