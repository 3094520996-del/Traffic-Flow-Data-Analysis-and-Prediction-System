#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据预处理脚本
对原始交通流数据进行清洗、特征工程和数据划分
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_raw_data():
    """加载原始数据"""
    print("=" * 60)
    print("开始数据预处理...")
    print("=" * 60)

    input_file = 'data/raw/traffic_data_raw.csv'
    print(f"\n正在读取数据：{input_file}")

    df = pd.read_csv(input_file, encoding='utf-8-sig')
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"数据加载完成：{len(df):,} 条记录")
    print(f"数据形状：{df.shape}")

    return df

def data_cleaning(df):
    """数据清洗"""
    print("\n" + "-" * 60)
    print("步骤1：数据清洗")
    print("-" * 60)

    # 1. 检查缺失值
    print("\n缺失值统计：")
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # 2. 填充缺失值（使用同路段、同时段的平均值）
    print("\n正在填充缺失值...")
    for col in ['speed', 'flow', 'occupancy']:
        if df[col].isnull().sum() > 0:
            df[col] = df.groupby(['road_id', 'hour'])[col].transform(
                lambda x: x.fillna(x.mean())
            )

    # 如果还有缺失值，用整体均值填充
    df.fillna(df.mean(numeric_only=True), inplace=True)

    print(f"OK 缺失值填充完成，剩余缺失值：{df.isnull().sum().sum()}")

    # 3. 异常值检测与处理
    print("\n正在处理异常值...")

    # 使用IQR方法检测flow的异常值
    Q1 = df.groupby('road_id')['flow'].transform('quantile', 0.25)
    Q3 = df.groupby('road_id')['flow'].transform('quantile', 0.75)
    IQR = Q3 - Q1

    # 标记异常值
    outlier_mask = (df['flow'] < (Q1 - 1.5 * IQR)) | (df['flow'] > (Q3 + 1.5 * IQR))
    outlier_count = outlier_mask.sum()

    print(f"检测到异常值：{outlier_count} 条 ({outlier_count/len(df)*100:.2f}%)")

    # 将异常值替换为分组中位数
    median_values = df.groupby('road_id')['flow'].transform('median')
    df.loc[outlier_mask, 'flow'] = median_values[outlier_mask].astype(int)

    print(f"OK 异常值处理完成")

    return df

def feature_engineering(df):
    """特征工程"""
    print("\n" + "-" * 60)
    print("步骤2：特征工程")
    print("-" * 60)

    # 1. 时间特征
    print("\n正在提取时间特征...")
    df['month'] = df['timestamp'].dt.month
    df['day'] = df['timestamp'].dt.day
    df['weekday'] = df['timestamp'].dt.weekday
    df['is_holiday'] = 0  # 简化处理，可根据实际情况标注节假日

    # 时段特征（0-凌晨, 1-早高峰, 2-平峰, 3-晚高峰, 4-夜间）
    def get_time_period(hour):
        if 7 <= hour < 9:
            return 1  # 早高峰
        elif 9 <= hour < 17:
            return 2  # 平峰
        elif 17 <= hour < 19:
            return 3  # 晚高峰
        elif 22 <= hour or hour < 6:
            return 4  # 夜间
        else:
            return 0  # 凌晨/其他

    df['time_period'] = df['hour'].apply(get_time_period)

    # 2. 拥堵指数（自定义公式）
    print("正在计算拥堵指数...")
    # 拥堵指数 = (标准速度 - 实际速度) / 标准速度 * 100
    road_standard_speed = df.groupby('road_id')['speed'].transform('quantile', 0.9)
    df['congestion_index'] = ((road_standard_speed - df['speed']) / road_standard_speed * 100).clip(0, 100)

    # 3. 历史特征（滞后特征）
    print("正在计算历史特征...")
    df = df.sort_values(['road_id', 'timestamp'])

    # 过去1小时的平均流量（12个5分钟记录）
    df['flow_last_1h'] = df.groupby('road_id')['flow'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )

    # 过去1小时的平均速度
    df['speed_last_1h'] = df.groupby('road_id')['speed'].transform(
        lambda x: x.rolling(window=12, min_periods=1).mean()
    )

    # 4. 空间特征
    print("正在编码空间特征...")
    # 区域编码
    df['area_code'] = pd.Categorical(df['area']).codes

    # 路段编码
    df['road_code'] = pd.Categorical(df['road_id']).codes

    # 5. 拥堵等级编码
    congestion_map = {'畅通': 0, '缓行': 1, '拥堵': 2, '严重拥堵': 3}
    df['congestion_code'] = df['congestion_level'].map(congestion_map)

    print(f"OK 特征工程完成，当前特征数：{df.shape[1]}")
    print(f"\n新增特征列表：")
    new_features = ['month', 'day', 'weekday', 'time_period', 'congestion_index',
                    'flow_last_1h', 'speed_last_1h', 'area_code', 'road_code', 'congestion_code']
    for feat in new_features:
        print(f"  - {feat}")

    return df

def prepare_modeling_data(df):
    """准备建模数据"""
    print("\n" + "-" * 60)
    print("步骤3：准备建模数据")
    print("-" * 60)

    # 1. 选择特征
    feature_cols = [
        'hour', 'minute', 'day_of_week', 'is_weekend', 'month', 'day',
        'weekday', 'time_period', 'occupancy', 'congestion_index',
        'flow_last_1h', 'speed_last_1h', 'area_code', 'road_code'
    ]

    target_col = 'flow'  # 预测目标：流量
    target_classification = 'congestion_code'  # 分类目标：拥堵等级

    print(f"\n特征数量：{len(feature_cols)}")
    print(f"预测目标（回归）：{target_col}")
    print(f"预测目标（分类）：{target_classification}")

    # 2. 提取特征和目标
    X = df[feature_cols].copy()
    y_regression = df[target_col].copy()
    y_classification = df[target_classification].copy()

    # 3. 数据划分（70% 训练，15% 验证，15% 测试）
    print("\n正在划分数据集...")

    # 先划分出测试集
    X_temp, X_test, y_reg_temp, y_reg_test, y_clf_temp, y_clf_test = train_test_split(
        X, y_regression, y_classification,
        test_size=0.15,
        random_state=42,
        shuffle=True
    )

    # 再从剩余数据中划分训练集和验证集
    X_train, X_val, y_reg_train, y_reg_val, y_clf_train, y_clf_val = train_test_split(
        X_temp, y_reg_temp, y_clf_temp,
        test_size=0.176,  # 0.176 ≈ 0.15 / 0.85，使验证集占总数据的15%
        random_state=42,
        shuffle=True
    )

    print(f"\n数据集划分完成：")
    print(f"  - 训练集：{len(X_train):,} 条 ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  - 验证集：{len(X_val):,} 条 ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  - 测试集：{len(X_test):,} 条 ({len(X_test)/len(X)*100:.1f}%)")

    # 4. 特征标准化
    print("\n正在标准化特征...")
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 转换回DataFrame（保留列名）
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=feature_cols, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)

    print(f"OK 特征标准化完成")

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_reg_train': y_reg_train,
        'y_reg_val': y_reg_val,
        'y_reg_test': y_reg_test,
        'y_clf_train': y_clf_train,
        'y_clf_val': y_clf_val,
        'y_clf_test': y_clf_test,
        'scaler': scaler,
        'feature_cols': feature_cols
    }

def save_processed_data(df, modeling_data):
    """保存处理后的数据"""
    print("\n" + "-" * 60)
    print("步骤4：保存数据")
    print("-" * 60)

    output_dir = 'data/processed'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 保存完整的处理后数据
    output_file = os.path.join(output_dir, 'traffic_data_processed.csv')
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nOK 完整数据已保存：{output_file}")

    # 2. 保存建模数据
    modeling_data['X_train'].to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    modeling_data['X_val'].to_csv(os.path.join(output_dir, 'X_val.csv'), index=False)
    modeling_data['X_test'].to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)

    modeling_data['y_reg_train'].to_csv(os.path.join(output_dir, 'y_reg_train.csv'), index=False)
    modeling_data['y_reg_val'].to_csv(os.path.join(output_dir, 'y_reg_val.csv'), index=False)
    modeling_data['y_reg_test'].to_csv(os.path.join(output_dir, 'y_reg_test.csv'), index=False)

    modeling_data['y_clf_train'].to_csv(os.path.join(output_dir, 'y_clf_train.csv'), index=False)
    modeling_data['y_clf_val'].to_csv(os.path.join(output_dir, 'y_clf_val.csv'), index=False)
    modeling_data['y_clf_test'].to_csv(os.path.join(output_dir, 'y_clf_test.csv'), index=False)

    print(f"OK 建模数据已保存：{output_dir}/")

    # 3. 保存Scaler
    scaler_file = os.path.join(output_dir, 'scaler.pkl')
    joblib.dump(modeling_data['scaler'], scaler_file)
    print(f"OK Scaler已保存：{scaler_file}")

    # 4. 保存特征列表
    feature_file = os.path.join(output_dir, 'feature_columns.txt')
    with open(feature_file, 'w', encoding='utf-8') as f:
        for col in modeling_data['feature_cols']:
            f.write(f"{col}\n")
    print(f"OK 特征列表已保存：{feature_file}")

def main():
    """主函数"""
    # 1. 加载数据
    df = load_raw_data()

    # 2. 数据清洗
    df = data_cleaning(df)

    # 3. 特征工程
    df = feature_engineering(df)

    # 4. 准备建模数据
    modeling_data = prepare_modeling_data(df)

    # 5. 保存数据
    save_processed_data(df, modeling_data)

    # 6. 数据质量报告
    print("\n" + "=" * 60)
    print("数据预处理完成！质量报告：")
    print("=" * 60)
    print(f"\n处理后数据统计：")
    print(f"  - 总记录数：{len(df):,}")
    print(f"  - 特征数量：{len(modeling_data['feature_cols'])}")
    print(f"  - 缺失值数量：{df.isnull().sum().sum()}")
    print(f"  - 时间范围：{df['timestamp'].min()} 至 {df['timestamp'].max()}")

    print(f"\n关键指标统计：")
    print(df[['flow', 'speed', 'occupancy', 'congestion_index']].describe())

    print("\n" + "=" * 60)
    print("OK 步骤2完成！请继续运行：python 3_data_analysis.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
