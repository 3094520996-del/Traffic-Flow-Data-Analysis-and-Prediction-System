#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交通流数据分析与预测 - Streamlit Web应用
提供交互式数据查询、可视化和预测功能
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from matplotlib import rcParams
from datetime import datetime, timedelta

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
rcParams['axes.unicode_minus'] = False

# 页面配置
st.set_page_config(
    page_title="交通流数据分析与预测系统",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """加载数据"""
    try:
        df = pd.read_csv('data/processed/traffic_data_processed.csv', encoding='utf-8-sig')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        st.error(f"数据加载失败：{e}")
        return None

@st.cache_resource
def load_models():
    """加载训练好的模型"""
    models = {}
    model_dir = 'outputs/models'

    try:
        models['Linear Regression'] = joblib.load(f'{model_dir}/linear_regression.pkl')
        models['Random Forest'] = joblib.load(f'{model_dir}/random_forest.pkl')
        models['XGBoost'] = joblib.load(f'{model_dir}/xgboost.pkl')
        scaler = joblib.load('data/processed/scaler.pkl')
        return models, scaler
    except Exception as e:
        st.warning(f"模型加载失败：{e}")
        return None, None

def main():
    """主函数"""

    # 标题
    st.markdown('<h1 class="main-header">🚗 交通流数据分析与预测系统</h1>', unsafe_allow_html=True)

    # 加载数据
    df = load_data()
    if df is None:
        st.error("请先运行数据生成和预处理脚本！")
        return

    models, scaler = load_models()

    # 侧边栏
    st.sidebar.title("📊 功能导航")
    page = st.sidebar.radio(
        "选择功能模块",
        ["数据概览", "数据分析", "流量预测", "拥堵分析", "关于项目"]
    )

    # 路由到不同页面
    if page == "数据概览":
        show_data_overview(df)
    elif page == "数据分析":
        show_data_analysis(df)
    elif page == "流量预测":
        show_prediction(df, models, scaler)
    elif page == "拥堵分析":
        show_congestion_analysis(df)
    elif page == "关于项目":
        show_about()

def show_data_overview(df):
    """数据概览页面"""
    st.header("📋 数据概览")

    # 关键指标
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("总记录数", f"{len(df):,}")
    with col2:
        st.metric("路段数量", f"{df['road_id'].nunique()}")
    with col3:
        st.metric("覆盖区域", f"{df['area'].nunique()}")
    with col4:
        st.metric("数据天数", f"{(df['timestamp'].max() - df['timestamp'].min()).days}")

    st.markdown("---")

    # 数据预览
    st.subheader("📊 数据预览")
    st.dataframe(df.head(100), use_container_width=True)

    # 基本统计
    st.subheader("📈 基本统计信息")
    col1, col2 = st.columns(2)

    with col1:
        st.write("**流量统计**")
        flow_stats = df['flow'].describe()
        st.dataframe(flow_stats, use_container_width=True)

    with col2:
        st.write("**速度统计**")
        speed_stats = df['speed'].describe()
        st.dataframe(speed_stats, use_container_width=True)

def show_data_analysis(df):
    """数据分析页面"""
    st.header("📊 数据分析")

    # 时间维度分析
    st.subheader("⏰ 时间维度分析")

    # 24小时流量变化
    hourly_flow = df.groupby('hour')['flow'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hourly_flow.index, hourly_flow.values, marker='o', linewidth=2, markersize=8)
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('平均流量（车辆/小时）', fontsize=12)
    ax.set_title('24小时交通流量变化趋势', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

    # 工作日vs周末
    col1, col2 = st.columns(2)

    with col1:
        weekday_flow = df[df['is_weekend'] == 0].groupby('hour')['flow'].mean()
        weekend_flow = df[df['is_weekend'] == 1].groupby('hour')['flow'].mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(weekday_flow.index, weekday_flow.values, marker='o', label='工作日', linewidth=2)
        ax.plot(weekend_flow.index, weekend_flow.values, marker='s', label='周末', linewidth=2)
        ax.set_xlabel('小时', fontsize=11)
        ax.set_ylabel('平均流量', fontsize=11)
        ax.set_title('工作日与周末对比', fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)

    with col2:
        # 拥堵等级分布
        congestion_counts = df['congestion_level'].value_counts()

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = ['#2ecc71', '#f39c12', '#e74c3c', '#c0392b']
        ax.pie(congestion_counts.values, labels=congestion_counts.index,
               autopct='%1.1f%%', colors=colors, startangle=90)
        ax.set_title('拥堵等级分布', fontsize=13, fontweight='bold')
        st.pyplot(fig)

    st.markdown("---")

    # 空间维度分析
    st.subheader("🗺️ 空间维度分析")

    # 路段选择
    selected_road = st.selectbox("选择路段", df['road_name'].unique())

    road_data = df[df['road_name'] == selected_road]

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("平均流量", f"{road_data['flow'].mean():.0f} 车辆/小时")
    with col2:
        st.metric("平均速度", f"{road_data['speed'].mean():.1f} km/h")
    with col3:
        st.metric("拥堵指数", f"{road_data['congestion_index'].mean():.1f}")

    # 该路段的24小时流量
    road_hourly = road_data.groupby('hour')['flow'].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(road_hourly.index, road_hourly.values, color='steelblue', edgecolor='black')
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('平均流量', fontsize=12)
    ax.set_title(f'{selected_road} - 24小时流量分布', fontsize=14, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3)
    st.pyplot(fig)

def show_prediction(df, models, scaler):
    """流量预测页面"""
    st.header("🔮 流量预测")

    if models is None or scaler is None:
        st.warning("模型未加载，请先运行模型训练脚本！")
        return

    st.subheader("📝 输入预测参数")

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_road = st.selectbox("路段", df['road_name'].unique(), key='pred_road')
        hour = st.slider("小时", 0, 23, 8)
        minute = st.slider("分钟", 0, 55, 0, step=5)

    with col2:
        day_of_week = st.selectbox("星期",
                                    ['周一', '周二', '周三', '周四', '周五', '周六', '周日'],
                                    index=0)
        is_weekend = 1 if day_of_week in ['周六', '周日'] else 0
        month = st.slider("月份", 1, 12, 6)
        day = st.slider("日期", 1, 31, 15)

    with col3:
        model_choice = st.selectbox("选择模型", list(models.keys()))

    # 准备特征
    road_id = df[df['road_name'] == selected_road]['road_id'].iloc[0]
    area = df[df['road_name'] == selected_road]['area'].iloc[0]

    # 获取该路段的历史平均数据作为基准
    road_hist = df[df['road_id'] == road_id]
    avg_occupancy = road_hist['occupancy'].mean()
    avg_congestion_index = road_hist['congestion_index'].mean()
    avg_flow_last_1h = road_hist['flow_last_1h'].mean()
    avg_speed_last_1h = road_hist['speed_last_1h'].mean()

    # 时段特征
    def get_time_period(h):
        if 7 <= h < 9:
            return 1
        elif 9 <= h < 17:
            return 2
        elif 17 <= h < 19:
            return 3
        elif 22 <= h or h < 6:
            return 4
        else:
            return 0

    time_period = get_time_period(hour)
    weekday = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'].index(day_of_week)
    area_code = df[df['area'] == area]['area_code'].iloc[0]
    road_code = df[df['road_id'] == road_id]['road_code'].iloc[0]

    # 构建特征向量
    features = pd.DataFrame([[
        hour, minute, weekday, is_weekend, month, day,
        weekday, time_period, avg_occupancy, avg_congestion_index,
        avg_flow_last_1h, avg_speed_last_1h, area_code, road_code
    ]], columns=[
        'hour', 'minute', 'day_of_week', 'is_weekend', 'month', 'day',
        'weekday', 'time_period', 'occupancy', 'congestion_index',
        'flow_last_1h', 'speed_last_1h', 'area_code', 'road_code'
    ])

    # 标准化
    features_scaled = scaler.transform(features)

    # 预测
    if st.button("> 开始预测", type="primary"):
        with st.spinner("正在预测..."):
            model = models[model_choice]
            prediction = model.predict(features_scaled)[0]

            st.success("预测完成！")

            # 显示结果
            st.markdown("---")
            st.subheader("📊 预测结果")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("预测流量", f"{prediction:.0f} 车辆/小时")

            with col2:
                # 判断拥堵等级
                if prediction < 1000:
                    congestion = "畅通 🟢"
                elif prediction < 1500:
                    congestion = "缓行 🟡"
                elif prediction < 2000:
                    congestion = "拥堵 🟠"
                else:
                    congestion = "严重拥堵 🔴"
                st.metric("预测拥堵状态", congestion)

            with col3:
                st.metric("使用模型", model_choice)

            # 与历史平均对比
            st.markdown("---")
            st.subheader("📈 与历史数据对比")

            hist_same_time = road_hist[road_hist['hour'] == hour]['flow'].mean()

            fig, ax = plt.subplots(figsize=(10, 5))
            categories = ['历史平均', '当前预测']
            values = [hist_same_time, prediction]
            colors = ['skyblue', 'coral']

            bars = ax.bar(categories, values, color=colors, edgecolor='black', width=0.5)
            ax.set_ylabel('流量（车辆/小时）', fontsize=12)
            ax.set_title(f'{selected_road} - 流量对比', fontsize=14, fontweight='bold')
            ax.grid(True, axis='y', alpha=0.3)

            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.0f}', ha='center', va='bottom', fontsize=11)

            st.pyplot(fig)

def show_congestion_analysis(df):
    """拥堵分析页面"""
    st.header("🚦 拥堵分析")

    # Top拥堵路段
    st.subheader("🔴 Top5 拥堵路段")

    top_congestion = df.groupby('road_name')['congestion_index'].mean().sort_values(ascending=False).head(5)

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(range(len(top_congestion)), top_congestion.values, color='#e74c3c', edgecolor='black')
    ax.set_yticks(range(len(top_congestion)))
    ax.set_yticklabels(top_congestion.index, fontsize=11)
    ax.set_xlabel('平均拥堵指数', fontsize=12)
    ax.set_title('Top5 拥堵路段', fontsize=14, fontweight='bold')
    ax.grid(True, axis='x', alpha=0.3)

    for i, (bar, value) in enumerate(zip(bars, top_congestion.values)):
        ax.text(value, i, f' {value:.1f}', va='center', fontsize=10)

    st.pyplot(fig)

    st.markdown("---")

    # 高峰时段拥堵分析
    st.subheader("⏰ 高峰时段拥堵热力图")

    pivot_data = df.pivot_table(values='congestion_index',
                                 index='road_name',
                                 columns='hour',
                                 aggfunc='mean')

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot_data, cmap='YlOrRd', annot=False, fmt='.0f',
                cbar_kws={'label': '拥堵指数'}, linewidths=0.5, ax=ax)
    ax.set_xlabel('小时', fontsize=12)
    ax.set_ylabel('路段', fontsize=12)
    ax.set_title('路段拥堵热力图', fontsize=14, fontweight='bold')
    st.pyplot(fig)

def show_about():
    """关于项目页面"""
    st.header("ℹ️ 关于项目")

    st.markdown("""
    ### 项目简介

    本项目是一个基于Python的**城市交通流数据分析与拥堵预测系统**，完整展示了数据科学项目的全流程：

    - 📊 **数据生成/采集**：模拟生成30天交通流数据
    - 🔧 **数据预处理**：清洗、特征工程、数据划分
    - 📈 **探索性分析**：时空维度分析、可视化
    - 🤖 **机器学习建模**：多模型对比（线性回归、随机森林、XGBoost）
    - 🌐 **Web应用开发**：Streamlit交互式应用

    ---

    ### 技术栈

    - **数据处理**：Pandas, NumPy
    - **数据可视化**：Matplotlib, Seaborn, Folium
    - **机器学习**：Scikit-learn, XGBoost
    - **Web框架**：Streamlit
    - **开发环境**：Python 3.8+

    ---

    ### 主要功能

    OK 交通流数据的时间和空间维度分析
    OK 多种可视化图表（流量趋势、拥堵热力图等）
    OK 基于机器学习的流量预测
    OK 拥堵路段识别与分析
    OK 交互式Web应用界面

    ---

    ### 项目成果

    - 处理并分析 **45万+** 条交通流数据
    - 生成 **15类** 数据分析图表
    - 训练 **3个** 机器学习模型
    - 预测精度：MAE < 15车辆/小时，R2 > 0.86
    - 识别 **5个** 核心拥堵区域

    ---

    ### 作者信息

    **姓名**：荚高伟
    **学校**：重庆交通大学
    **专业**：智慧交通
    **邮箱**：13651562565@163.com

    ---

    ### 项目地址

    📂 项目代码：[GitHub链接]
    📄 技术文档：查看项目README.md

    """)

if __name__ == '__main__':
    main()
