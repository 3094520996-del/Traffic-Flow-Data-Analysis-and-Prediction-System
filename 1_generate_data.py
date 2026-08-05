#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通流数据生成脚本
模拟生成重庆主城区30天的交通流数据
包含：时间、路段、流量、速度、占有率等信息
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# 设置随机种子，保证结果可复现
np.random.seed(42)

def generate_traffic_data():
    """
    生成模拟交通流数据
    """
    print("=" * 60)
    print("开始生成交通流数据...")
    print("=" * 60)

    # 1. 基本参数设置
    start_date = datetime(2026, 6, 1, 0, 0, 0)
    days = 30  # 生成30天数据
    time_interval = 5  # 每5分钟一条记录
    records_per_day = 24 * 60 // time_interval  # 每天288条记录

    # 2. 路段信息
    road_segments = [
        # 格式：(路段ID, 路段名称, 区域, 基础流量, 基础速度)
        ('R001', '解放碑至朝天门大桥', '渝中区', 1200, 45),
        ('R002', '观音桥至红旗河沟', '江北区', 1500, 50),
        ('R003', '杨家坪至石桥铺', '九龙坡区', 1100, 48),
        ('R004', '南坪至四公里', '南岸区', 1300, 46),
        ('R005', '沙坪坝至大学城', '沙坪坝区', 1400, 52),
        ('R006', '江北机场高速', '渝北区', 1600, 80),
        ('R007', '内环快速路东段', '南岸区', 1800, 60),
        ('R008', '内环快速路西段', '九龙坡区', 1700, 58),
        ('R009', '渝澳大桥', '渝中区', 1000, 40),
        ('R010', '菜园坝大桥', '渝中区', 900, 38),
        ('R011', '北碚至缙云山', '北碚区', 800, 55),
        ('R012', '巴南区鱼洞至界石', '巴南区', 950, 50),
        ('R013', '两路至空港', '渝北区', 1250, 60),
        ('R014', '大渡口至九龙坡', '大渡口区', 1050, 45),
        ('R015', '南滨路', '南岸区', 1100, 42),
    ]

    # 3. 生成时间序列
    print(f"\n正在生成 {days} 天的时间序列...")
    time_list = []
    for day in range(days):
        for record in range(records_per_day):
            timestamp = start_date + timedelta(days=day, minutes=record * time_interval)
            time_list.append(timestamp)

    print(f"时间序列生成完成：{len(time_list)} 个时间点")

    # 4. 为每个路段生成交通流数据
    all_data = []

    for seg_id, seg_name, area, base_flow, base_speed in road_segments:
        print(f"\n正在生成路段数据：{seg_name} ({seg_id})")

        for timestamp in time_list:
            # 提取时间特征
            hour = timestamp.hour
            minute = timestamp.minute
            day_of_week = timestamp.weekday()  # 0=周一, 6=周日
            is_weekend = 1 if day_of_week >= 5 else 0

            # 5. 流量模型（考虑早晚高峰）
            # 基础流量
            flow = base_flow

            # 早高峰（7:00-9:00）
            if 7 <= hour < 9:
                flow *= (1.5 + 0.3 * np.sin((hour - 7) * np.pi / 2))
            # 晚高峰（17:00-19:00）
            elif 17 <= hour < 19:
                flow *= (1.6 + 0.4 * np.sin((hour - 17) * np.pi / 2))
            # 平峰
            elif 9 <= hour < 17:
                flow *= 1.1
            # 夜间（0:00-6:00, 22:00-24:00）
            else:
                flow *= 0.4

            # 周末流量衰减
            if is_weekend:
                flow *= 0.75

            # 添加随机波动
            flow *= (1 + np.random.normal(0, 0.15))
            flow = max(0, int(flow))  # 确保非负

            # 6. 速度模型（流量越大，速度越慢）
            # 使用简化的速度-流量关系
            if flow < base_flow * 0.8:
                speed = base_speed
            elif flow < base_flow * 1.2:
                speed = base_speed * 0.85
            elif flow < base_flow * 1.5:
                speed = base_speed * 0.65
            else:
                speed = base_speed * 0.45

            # 添加随机波动
            speed *= (1 + np.random.normal(0, 0.1))
            speed = max(10, min(speed, base_speed * 1.2))  # 限制速度范围

            # 7. 占有率（车辆占据道路的比例，%）
            # 简化模型：流量越大，占有率越高
            occupancy = (flow / (base_flow * 2)) * 100
            occupancy = max(5, min(occupancy, 95))  # 限制在5%-95%

            # 8. 拥堵状态（基于速度判断）
            if speed >= base_speed * 0.8:
                congestion_level = '畅通'
            elif speed >= base_speed * 0.6:
                congestion_level = '缓行'
            elif speed >= base_speed * 0.4:
                congestion_level = '拥堵'
            else:
                congestion_level = '严重拥堵'

            # 9. 构建记录
            record = {
                'timestamp': timestamp,
                'date': timestamp.date(),
                'time': timestamp.time(),
                'hour': hour,
                'minute': minute,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'road_id': seg_id,
                'road_name': seg_name,
                'area': area,
                'flow': flow,  # 车辆数/小时
                'speed': round(speed, 1),  # km/h
                'occupancy': round(occupancy, 1),  # %
                'congestion_level': congestion_level,
            }

            all_data.append(record)

    # 10. 转换为DataFrame
    print(f"\n正在整合数据...")
    df = pd.DataFrame(all_data)

    # 11. 添加一些缺失值和异常值（模拟真实数据）
    print("添加缺失值和异常值...")
    missing_indices = np.random.choice(df.index, size=int(len(df) * 0.02), replace=False)
    df.loc[missing_indices, 'speed'] = np.nan

    outlier_indices = np.random.choice(df.index, size=int(len(df) * 0.01), replace=False)
    df.loc[outlier_indices, 'flow'] = df.loc[outlier_indices, 'flow'] * 3

    # 12. 保存数据
    output_dir = 'data/raw'
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'traffic_data_raw.csv')

    df.to_csv(output_file, index=False, encoding='utf-8-sig')

    # 13. 数据统计
    print("\n" + "=" * 60)
    print("数据生成完成！")
    print("=" * 60)
    print(f"\n数据统计信息：")
    print(f"  - 记录总数：{len(df):,} 条")
    print(f"  - 时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")
    print(f"  - 路段数量：{df['road_id'].nunique()} 个")
    print(f"  - 覆盖区域：{df['area'].nunique()} 个")
    print(f"  - 缺失值数量：{df['speed'].isna().sum()} 条 ({df['speed'].isna().sum()/len(df)*100:.2f}%)")
    print(f"\n保存路径：{output_file}")

    print(f"\n数据预览：")
    print(df.head(10))

    print(f"\n数据描述：")
    print(df[['flow', 'speed', 'occupancy']].describe())

    print("\n" + "=" * 60)
    print("OK 步骤1完成！请继续运行：python 2_data_preprocessing.py")
    print("=" * 60)

    return df

if __name__ == '__main__':
    df = generate_traffic_data()
