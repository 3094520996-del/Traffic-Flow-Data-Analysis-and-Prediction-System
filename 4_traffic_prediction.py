#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通流预测模型训练脚本
使用多种机器学习算法进行流量预测，并对比模型性能
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

def load_data():
    """加载训练数据"""
    print("=" * 60)
    print("开始训练预测模型...")
    print("=" * 60)

    data_dir = 'data/processed'

    print(f"\n正在加载训练数据...")
    X_train = pd.read_csv(f'{data_dir}/X_train.csv')
    X_val = pd.read_csv(f'{data_dir}/X_val.csv')
    X_test = pd.read_csv(f'{data_dir}/X_test.csv')

    y_train = pd.read_csv(f'{data_dir}/y_reg_train.csv').values.ravel()
    y_val = pd.read_csv(f'{data_dir}/y_reg_val.csv').values.ravel()
    y_test = pd.read_csv(f'{data_dir}/y_reg_test.csv').values.ravel()

    print(f"OK 数据加载完成")
    print(f"  训练集：{len(X_train):,} 条")
    print(f"  验证集：{len(X_val):,} 条")
    print(f"  测试集：{len(X_test):,} 条")

    return X_train, X_val, X_test, y_train, y_val, y_test

def train_linear_regression(X_train, y_train, X_val, y_val):
    """训练线性回归模型（基线模型）"""
    print("\n" + "-" * 60)
    print("模型1：线性回归（基线模型）")
    print("-" * 60)

    print("正在训练...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 评估
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n训练集性能：")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R2:   {train_r2:.4f}")

    print(f"\n验证集性能：")
    print(f"  MAE:  {val_mae:.2f}")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  R2:   {val_r2:.4f}")

    return model, {
        'name': 'Linear Regression',
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_pred': y_val_pred
    }

def train_random_forest(X_train, y_train, X_val, y_val):
    """训练随机森林模型"""
    print("\n" + "-" * 60)
    print("模型2：随机森林")
    print("-" * 60)

    print("正在训练...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )
    model.fit(X_train, y_train)

    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 评估
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n训练集性能：")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R2:   {train_r2:.4f}")

    print(f"\n验证集性能：")
    print(f"  MAE:  {val_mae:.2f}")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  R2:   {val_r2:.4f}")

    return model, {
        'name': 'Random Forest',
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_pred': y_val_pred,
        'feature_importance': model.feature_importances_
    }

def train_xgboost(X_train, y_train, X_val, y_val):
    """训练XGBoost模型"""
    print("\n" + "-" * 60)
    print("模型3：XGBoost")
    print("-" * 60)

    print("正在训练...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    # 预测
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)

    # 评估
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)

    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)

    print(f"\n训练集性能：")
    print(f"  MAE:  {train_mae:.2f}")
    print(f"  RMSE: {train_rmse:.2f}")
    print(f"  R2:   {train_r2:.4f}")

    print(f"\n验证集性能：")
    print(f"  MAE:  {val_mae:.2f}")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  R2:   {val_r2:.4f}")

    return model, {
        'name': 'XGBoost',
        'train_mae': train_mae,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_mae': val_mae,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'val_pred': y_val_pred,
        'feature_importance': model.feature_importances_
    }

def evaluate_on_test_set(models, X_test, y_test):
    """在测试集上评估所有模型"""
    print("\n" + "=" * 60)
    print("最终测试集评估")
    print("=" * 60)

    test_results = []

    for model_name, model_obj in models.items():
        print(f"\n{model_name}:")
        y_pred = model_obj.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

        print(f"  MAE:  {mae:.2f}")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  R2:   {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")

        test_results.append({
            'model': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'predictions': y_pred
        })

    return test_results

def visualize_results(results, y_val, test_results, y_test, X_train):
    """可视化模型结果"""
    print("\n" + "-" * 60)
    print("生成可视化结果")
    print("-" * 60)

    output_dir = 'outputs/figures'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 模型性能对比（验证集）
    print("\n正在绘制模型性能对比图...")
    metrics = ['MAE', 'RMSE', 'R2']
    model_names = [r['name'] for r in results]

    mae_values = [r['val_mae'] for r in results]
    rmse_values = [r['val_rmse'] for r in results]
    r2_values = [r['val_r2'] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # MAE
    bars1 = axes[0].bar(model_names, mae_values, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0].set_ylabel('MAE', fontsize=12)
    axes[0].set_title('平均绝对误差（MAE）', fontsize=13, fontweight='bold')
    axes[0].grid(True, axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # RMSE
    bars2 = axes[1].bar(model_names, rmse_values, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1].set_ylabel('RMSE', fontsize=12)
    axes[1].set_title('均方根误差（RMSE）', fontsize=13, fontweight='bold')
    axes[1].grid(True, axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=10)

    # R2
    bars3 = axes[2].bar(model_names, r2_values, color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[2].set_ylabel('R2', fontsize=12)
    axes[2].set_title('决定系数（R2）', fontsize=13, fontweight='bold')
    axes[2].grid(True, axis='y', alpha=0.3)
    axes[2].set_ylim([min(r2_values) * 0.95, 1.0])
    for bar in bars3:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/12_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：12_model_comparison.png")

    # 2. 预测值vs真实值对比
    print("正在绘制预测值vs真实值对比图...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for idx, (result, ax) in enumerate(zip(results, axes)):
        sample_size = min(1000, len(y_val))
        indices = np.random.choice(len(y_val), sample_size, replace=False)

        ax.scatter(y_val[indices], result['val_pred'][indices],
                  alpha=0.5, s=20, edgecolors='black', linewidth=0.5)
        ax.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()],
               'r--', linewidth=2, label='理想预测')
        ax.set_xlabel('真实流量', fontsize=11)
        ax.set_ylabel('预测流量', fontsize=11)
        ax.set_title(f'{result["name"]}\n(R2={result["val_r2"]:.4f})',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/13_prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：13_prediction_vs_actual.png")

    # 3. 特征重要性（随机森林）
    if 'feature_importance' in results[1]:  # Random Forest
        print("正在绘制特征重要性图...")
        feature_names = X_train.columns
        importance = results[1]['feature_importance']

        # 排序
        indices = np.argsort(importance)[::-1][:10]  # Top 10

        plt.figure(figsize=(12, 6))
        plt.bar(range(len(indices)), importance[indices],
               color='steelblue', edgecolor='black')
        plt.xticks(range(len(indices)),
                  [feature_names[i] for i in indices],
                  rotation=45, ha='right', fontsize=10)
        plt.ylabel('重要性', fontsize=12)
        plt.title('随机森林特征重要性（Top 10）', fontsize=14, fontweight='bold')
        plt.grid(True, axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/14_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  OK 保存图表：14_feature_importance.png")

    # 4. 测试集性能总结
    print("正在绘制测试集性能总结...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # MAE
    mae_vals = [r['mae'] for r in test_results]
    axes[0, 0].bar([r['model'] for r in test_results], mae_vals,
                   color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0, 0].set_ylabel('MAE', fontsize=11)
    axes[0, 0].set_title('测试集 - 平均绝对误差', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, axis='y', alpha=0.3)

    # RMSE
    rmse_vals = [r['rmse'] for r in test_results]
    axes[0, 1].bar([r['model'] for r in test_results], rmse_vals,
                   color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[0, 1].set_ylabel('RMSE', fontsize=11)
    axes[0, 1].set_title('测试集 - 均方根误差', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, axis='y', alpha=0.3)

    # R2
    r2_vals = [r['r2'] for r in test_results]
    axes[1, 0].bar([r['model'] for r in test_results], r2_vals,
                   color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1, 0].set_ylabel('R2', fontsize=11)
    axes[1, 0].set_title('测试集 - 决定系数', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, axis='y', alpha=0.3)

    # MAPE
    mape_vals = [r['mape'] for r in test_results]
    axes[1, 1].bar([r['model'] for r in test_results], mape_vals,
                   color=['#3498db', '#2ecc71', '#e74c3c'], edgecolor='black')
    axes[1, 1].set_ylabel('MAPE (%)', fontsize=11)
    axes[1, 1].set_title('测试集 - 平均绝对百分比误差', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/15_test_performance_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  OK 保存图表：15_test_performance_summary.png")

def save_models(models):
    """保存训练好的模型"""
    print("\n" + "-" * 60)
    print("保存模型")
    print("-" * 60)

    model_dir = 'outputs/models'
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.pkl'
        filepath = os.path.join(model_dir, filename)
        joblib.dump(model, filepath)
        print(f"OK 保存模型：{filepath}")

def main():
    """主函数"""
    # 1. 加载数据
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    # 2. 训练模型
    lr_model, lr_result = train_linear_regression(X_train, y_train, X_val, y_val)
    rf_model, rf_result = train_random_forest(X_train, y_train, X_val, y_val)
    xgb_model, xgb_result = train_xgboost(X_train, y_train, X_val, y_val)

    results = [lr_result, rf_result, xgb_result]
    models = {
        'Linear Regression': lr_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

    # 3. 测试集评估
    test_results = evaluate_on_test_set(models, X_test, y_test)

    # 4. 可视化结果
    visualize_results(results, y_val, test_results, y_test, X_train)

    # 5. 保存模型
    save_models(models)

    # 6. 输出最佳模型
    print("\n" + "=" * 60)
    print("模型训练完成！")
    print("=" * 60)

    best_model_idx = np.argmin([r['val_mae'] for r in results])
    best_model = results[best_model_idx]

    print(f"\n🏆 最佳模型：{best_model['name']}")
    print(f"  验证集 MAE:  {best_model['val_mae']:.2f}")
    print(f"  验证集 RMSE: {best_model['val_rmse']:.2f}")
    print(f"  验证集 R2:   {best_model['val_r2']:.4f}")

    test_best = test_results[best_model_idx]
    print(f"\n  测试集 MAE:  {test_best['mae']:.2f}")
    print(f"  测试集 RMSE: {test_best['rmse']:.2f}")
    print(f"  测试集 R2:   {test_best['r2']:.4f}")
    print(f"  测试集 MAPE: {test_best['mape']:.2f}%")

    print("\n" + "=" * 60)
    print("OK 步骤4完成！请继续运行：streamlit run 5_streamlit_app.py")
    print("=" * 60)

if __name__ == '__main__':
    main()
