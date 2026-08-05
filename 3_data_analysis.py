#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据分析与可视化脚本
对交通流数据进行探索性分析，生成多类可视化图表
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 设置seaborn风格
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_data():
    """加载处理后的数据"""
    print("=" * 60)
    print("开始数据分析与可视化...")
    print("=" * 60)

    input_file = 'data/processed/traffic_data_processed.csv'
    print(f"\n正在读取数据：{input_file}")

    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"数据加载完成：{len(df):,} 条记录")

    return df

def create_output_dir():
    """创建输出目录"""
    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def analyze_time_patterns(df, output_dir):
    """时间维度分析"""
    print("\n" + "-" * 60)
    print("分析1：时间维度分析")
    print("-" * 60)

    # 1. 24小时流量变化
    print("\n正在绘制24小时流量变化曲线...")
    hourly_flow = df.groupby('hour')['flow'].agg(['mean', 'std'])

    plt.figure(figsize=(12, 6))
    plt.plot(hourly_flow.index, hourly_flow['mean'], marker='o', linewidth=2, markersize=8)
    plt.fill_between(hourly_flow.index,
                     hourly_flow['mean'] - hourly_flow['std'],
                     hourly_flow['mean'] + hourly_flow['std'],
                     alpha=0.3)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('平均流量（车辆/小时）', fontsize=12)
    plt.title('24小时交通流量变化趋势', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_hourly_flow_pattern.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：01_hourly_flow_pattern.png")

    # 2. 工作日vs周末对比
    print("正在绘制工作日vs周末对比图...")
    weekday_flow = df[df['is_weekend'] == 0].groupby('hour')['flow'].mean()
    weekend_flow = df[df['is_weekend'] == 1].groupby('hour')['flow'].mean()

    plt.figure(figsize=(12, 6))
    plt.plot(weekday_flow.index, weekday_flow.values, marker='o', label='工作日', linewidth=2)
    plt.plot(weekend_flow.index, weekend_flow.values, marker='s', label='周末', linewidth=2)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('平均流量（车辆/小时）', fontsize=12)
    plt.title('工作日与周末交通流量对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_weekday_vs_weekend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：02_weekday_vs_weekend.png")

    # 3. 一周流量分布
    print("正在绘制一周流量分布图...")
    weekday_names = ['周一', '周二', '周三', '周四', '周五', '周六', '周日']
    weekly_flow = df.groupby('day_of_week')['flow'].mean()

    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(7), weekly_flow.values, color=sns.color_palette("husl", 7), edgecolor='black')
    plt.xticks(range(7), weekday_names, fontsize=11)
    plt.ylabel('平均流量（车辆/小时）', fontsize=12)
    plt.title('一周各天平均交通流量', fontsize=14, fontweight='bold')
    plt.grid(True, axis='y', alpha=0.3)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_weekly_flow_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：03_weekly_flow_distribution.png")

def analyze_spatial_patterns(df, output_dir):
    """空间维度分析"""
    print("\n" + "-" * 60)
    print("分析2：空间维度分析")
    print("-" * 60)

    # 4. 各路段平均流量对比
    print("\n正在绘制路段流量对比图...")
    road_flow = df.groupby('road_name')['flow'].mean().sort_values(ascending=False)

    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(road_flow)), road_flow.values, color=sns.color_palette("viridis", len(road_flow)))
    plt.yticks(range(len(road_flow)), road_flow.index, fontsize=10)
    plt.xlabel('平均流量（车辆/小时）', fontsize=12)
    plt.title('各路段平均交通流量对比', fontsize=14, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)

    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, road_flow.values)):
        plt.text(value, i, f' {int(value)}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_road_flow_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：04_road_flow_comparison.png")

    # 5. 各区域流量统计
    print("正在绘制区域流量统计图...")
    area_stats = df.groupby('area')['flow'].agg(['mean', 'median', 'std']).sort_values('mean', ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(area_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, area_stats['mean'], width, label='平均值', color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, area_stats['median'], width, label='中位数', color='lightcoral', edgecolor='black')

    ax.set_xlabel('区域', fontsize=12)
    ax.set_ylabel('流量（车辆/小时）', fontsize=12)
    ax.set_title('各区域交通流量统计', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(area_stats.index, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_area_flow_statistics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：05_area_flow_statistics.png")

    # 6. 拥堵热力图（路段 × 时间）
    print("正在绘制拥堵热力图...")
    pivot_data = df.pivot_table(values='congestion_index',
                                 index='road_name',
                                 columns='hour',
                                 aggfunc='mean')

    plt.figure(figsize=(16, 10))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='.0f',
                cbar_kws={'label': '拥堵指数'}, linewidths=0.5)
    plt.xlabel('小时', fontsize=12)
    plt.ylabel('路段', fontsize=12)
    plt.title('路段拥堵热力图（按小时）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/06_congestion_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：06_congestion_heatmap.png")

def analyze_traffic_characteristics(df, output_dir):
    """交通特性分析"""
    print("\n" + "-" * 60)
    print("分析3：交通特性分析")
    print("-" * 60)

    # 7. 速度-流量关系散点图
    print("\n正在绘制速度-流量关系图...")
    sample_df = df.sample(n=min(5000, len(df)), random_state=42)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(sample_df['flow'], sample_df['speed'],
                         c=sample_df['congestion_index'], cmap='RdYlGn_r',
                         alpha=0.6, s=30, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label='拥堵指数')
    plt.xlabel('流量（车辆/小时）', fontsize=12)
    plt.ylabel('速度（km/h）', fontsize=12)
    plt.title('速度-流量关系散点图', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/07_speed_flow_relationship.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：07_speed_flow_relationship.png")

    # 8. 流量分布直方图
    print("正在绘制流量分布直方图...")
    plt.figure(figsize=(12, 6))
    plt.hist(df['flow'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    plt.axvline(df['flow'].mean(), color='red', linestyle='--', linewidth=2, label=f"平均值: {df['flow'].mean():.0f}")
    plt.axvline(df['flow'].median(), color='green', linestyle='--', linewidth=2, label=f"中位数: {df['flow'].median():.0f}")
    plt.xlabel('流量（车辆/小时）', fontsize=12)
    plt.ylabel('频数', fontsize=12)
    plt.title('交通流量分布直方图', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/08_flow_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：08_flow_distribution.png")

    # 9. 拥堵等级占比
    print("正在绘制拥堵等级占比图...")
    congestion_counts = df['congestion_level'].value_counts()

    plt.figure(figsize=(10, 8))
    colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
    explode = (0.05, 0.05, 0.05, 0.05)

    plt.pie(congestion_counts.values, labels=congestion_counts.index,
            autopct='%1.1f%%', startangle=90, colors=colors, explode=explode,
            textprops={'fontsize': 12}, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    plt.title('交通拥堵等级分布', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/09_congestion_level_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：09_congestion_level_distribution.png")

    # 10. 相关性热力图
    print("正在绘制相关性热力图...")
    corr_cols = ['flow', 'speed', 'occupancy', 'congestion_index', 'hour', 'day_of_week']
    correlation = df[corr_cols].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('交通指标相关性矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/10_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：10_correlation_heatmap.png")

def analyze_peak_hours(df, output_dir):
    """高峰时段分析"""
    print("\n" + "-" * 60)
    print("分析4：高峰时段分析")
    print("-" * 60)

    # 11. 早晚高峰对比
    print("\n正在绘制早晚高峰对比图...")
    morning_peak = df[(df['hour'] >= 7) & (df['hour'] < 9)]
    evening_peak = df[(df['hour'] >= 17) & (df['hour'] < 19)]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 早高峰
    morning_road_flow = morning_peak.groupby('road_name')['flow'].mean().sort_values(ascending=False).head(10)
    axes[0].barh(range(len(morning_road_flow)), morning_road_flow.values,
                 color='orange', edgecolor='black')
    axes[0].set_yticks(range(len(morning_road_flow)))
    axes[0].set_yticklabels(morning_road_flow.index, fontsize=9)
    axes[0].set_xlabel('平均流量（车辆/小时）', fontsize=11)
    axes[0].set_title('早高峰（7:00-9:00）Top10路段', fontsize=12, fontweight='bold')
    axes[0].grid(True, axis='x', alpha=0.3)

    # 晚高峰
    evening_road_flow = evening_peak.groupby('road_name')['flow'].mean().sort_values(ascending=False).head(10)
    axes[1].barh(range(len(evening_road_flow)), evening_road_flow.values,
                 color='purple', edgecolor='black')
    axes[1].set_yticks(range(len(evening_road_flow)))
    axes[1].set_yticklabels(evening_road_flow.index, fontsize=9)
    axes[1].set_xlabel('平均流量（车辆/小时）', fontsize=11)
    axes[1].set_title('晚高峰（17:00-19:00）Top10路段', fontsize=12, fontweight='bold')
    axes[1].grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/11_peak_hours_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：11_peak_hours_comparison.png")

def generate_summary_report(df, output_dir):
    """生成数据分析摘要报告"""
    print("\n" + "-" * 60)
    print("生成数据分析摘要报告")
    print("-" * 60)

    report = []
    report.append("=" * 80)
    report.append("交通流数据分析报告")
    report.append("=" * 80)
    report.append("")

    # 1. 数据概况
    report.append("一、数据概况")
    report.append("-" * 80)
    report.append(f"数据记录总数：{len(df):,} 条")
    report.append(f"时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")
    report.append(f"路段数量：{df['road_id'].nunique()} 个")
    report.append(f"覆盖区域：{df['area'].nunique()} 个（{', '.join(df['area'].unique())}）")
    report.append("")

    # 2. 流量统计
    report.append("二、交通流量统计")
    report.append("-" * 80)
    report.append(f"平均流量：{df['flow'].mean():.1f} 车辆/小时")
    report.append(f"最大流量：{df['flow'].max():.0f} 车辆/小时")
    report.append(f"最小流量：{df['flow'].min():.0f} 车辆/小时")
    report.append(f"流量标准差：{df['flow'].std():.1f}")
    report.append("")

    # 3. 速度统计
    report.append("三、交通速度统计")
    report.append("-" * 80)
    report.append(f"平均速度：{df['speed'].mean():.1f} km/h")
    report.append(f"最大速度：{df['speed'].max():.1f} km/h")
    report.append(f"最小速度：{df['speed'].min():.1f} km/h")
    report.append("")

    # 4. 高峰时段识别
    report.append("四、高峰时段识别")
    report.append("-" * 80)
    hourly_flow = df.groupby('hour')['flow'].mean()
    morning_peak_hour = hourly_flow[7:9].idxmax()
    evening_peak_hour = hourly_flow[17:19].idxmax()
    report.append(f"早高峰时段：7:00-9:00（峰值：{morning_peak_hour}:00，流量：{hourly_flow[morning_peak_hour]:.0f}）")
    report.append(f"晚高峰时段：17:00-19:00（峰值：{evening_peak_hour}:00，流量：{hourly_flow[evening_peak_hour]:.0f}）")
    report.append("")

    # 5. 拥堵分析
    report.append("五、拥堵情况分析")
    report.append("-" * 80)
    congestion_counts = df['congestion_level'].value_counts()
    for level, count in congestion_counts.items():
        percentage = count / len(df) * 100
        report.append(f"{level}：{count:,} 条 ({percentage:.1f}%)")
    report.append("")

    # 6. 核心拥堵路段
    report.append("六、Top5 拥堵路段（按平均拥堵指数）")
    report.append("-" * 80)
    top_congestion_roads = df.groupby('road_name')['congestion_index'].mean().sort_values(ascending=False).head(5)
    for i, (road, index) in enumerate(top_congestion_roads.items(), 1):
        report.append(f"{i}. {road}：拥堵指数 {index:.1f}")
    report.append("")

    # 7. 工作日vs周末
    report.append("七、工作日与周末对比")
    report.append("-" * 80)
    weekday_avg = df[df['is_weekend'] == 0]['flow'].mean()
    weekend_avg = df[df['is_weekend'] == 1]['flow'].mean()
    report.append(f"工作日平均流量：{weekday_avg:.1f} 车辆/小时")
    report.append(f"周末平均流量：{weekend_avg:.1f} 车辆/小时")
    report.append(f"差异：{abs(weekday_avg - weekend_avg):.1f} ({abs(weekday_avg - weekend_avg) / weekday_avg * 100:.1f}%)")
    report.append("")

    report.append("=" * 80)
    report.append("报告生成完成")
    report.append("=" * 80)

    # 保存报告
    report_text = "\n".join(report)
    report_file = os.path.join(output_dir, '00_analysis_summary.txt')
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"\n{report_text}")
    print(f"\nOK 报告已保存：{report_file}")

def main():
    """主函数"""
    # 1. 加载数据
    df = load_data()

    # 2. 创建输出目录
    output_dir = create_output_dir()

    # 3. 时间维度分析
    analyze_time_patterns(df, output_dir)

    # 4. 空间维度分析
    analyze_spatial_patterns(df, output_dir)

    # 5. 交通特性分析
    analyze_traffic_characteristics(df, output_dir)

    # 6. 高峰时段分析
    analyze_peak_hours(df, output_dir)

    # 7. 生成摘要报告
    generate_summary_report(df, output_dir)

    print("\n" + "=" * 60)
    print("数据分析完成！")
    print("=" * 60)
    print(f"\nOK 共生成 11 类可视化图表")
    print(f"OK 图表保存位置：{output_dir}/")
    print(f"\n图表列表：")
    for i in range(1, 12):
        print(f"  {i:2d}. 查看 {output_dir}/ 中的图表文件")

    print("\n" + "=" * 60)
    print("OK 步骤3完成！请继续运行：python 4_traffic_prediction.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
