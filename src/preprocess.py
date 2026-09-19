# -*- coding: utf-8 -*-
"""
数据预处理脚本
功能：
1. 加载bank.csv数据集
2. 数据探索与统计分析
3. 处理缺失值和异常值
4. 类别变量编码
5. 保存处理后的数据供Spark Scala使用
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ==================== 路径配置 ====================
# 以仓库根目录（src 的上级目录）为基准，保证在任意工作目录下都能运行
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("银行营销数据集 - 数据预处理")
print("=" * 60)

# ==================== 1. 加载数据 ====================
print("\n[1] 加载数据...")
df = pd.read_csv(DATA_DIR / 'bank.csv', sep=';')
print(f"数据集大小: {df.shape[0]} 行 × {df.shape[1]} 列")

# ==================== 2. 数据探索 ====================
print("\n[2] 数据探索...")

# 检查缺失值
print(f"缺失值统计: {df.isnull().sum().sum()} 个缺失值")

# 检查"unknown"值在各列的分布
print("\n各列中 'unknown' 值的分布:")
for col in df.select_dtypes(include='object').columns:
    unknown_count = (df[col] == 'unknown').sum()
    if unknown_count > 0:
        print(f"  {col}: {unknown_count} ({unknown_count/len(df)*100:.2f}%)")

# 数值列的统计信息
print("\n数值列统计:")
numeric_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
for col in numeric_cols:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    print(f"  {col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, "
          f"min={df[col].min()}, max={df[col].max()}, 异常值(IQR)={outliers}")

# 目标变量分布
print(f"\n目标变量 'y' 分布:")
print(df['y'].value_counts())
print(f"比例: {df['y'].value_counts(normalize=True).to_dict()}")

# ==================== 3. 数据预处理 ====================
print("\n[3] 数据预处理...")

# 3.1 处理 duration = 0 的情况（表示未联系成功）
zero_duration = (df['duration'] == 0).sum()
print(f"duration=0 的记录数: {zero_duration}")

# 3.2 将 pdays=-1 转换为标记（-1表示之前没有联系过）
df['pdays_contacted'] = (df['pdays'] != -1).astype(int)
print(f"之前有联系过的客户数: {df['pdays_contacted'].sum()}")

# 3.3 创建年龄分组特征
df['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100],
                          labels=['<25', '25-35', '35-45', '45-55', '>55'])
print(f"年龄分组分布:\n{df['age_group'].value_counts()}")

# 3.4 创建余额分组特征
df['balance_group'] = pd.cut(df['balance'],
                              bins=[-float('inf'), 0, 500, 2000, 10000, float('inf')],
                              labels=['负余额', '低(0-500)', '中(500-2000)', '高(2000-10k)', '极高(>10k)'])
print(f"余额分组分布:\n{df['balance_group'].value_counts()}")

# 3.5 统计每个客户的联系总次数
df['total_contacts'] = df['campaign'] + df['previous']
print(f"总联系次数统计: mean={df['total_contacts'].mean():.2f}, "
      f"median={df['total_contacts'].median():.0f}, max={df['total_contacts'].max()}")

# ==================== 4. 数据保存 ====================
print("\n[4] 保存预处理后的数据...")

# 保存完整预处理数据（CSV格式，供Scala Spark读取）
output_file = OUTPUT_DIR / 'bank_processed.csv'
df.to_csv(output_file, index=False, sep=';')
print(f"预处理后数据已保存到: {output_file}")
print(f"总列数: {len(df.columns)}")

# 保存统计结果用于可视化
stats = {
    'total_rows': len(df),
    'total_cols': len(df.columns),
    'y_distribution': df['y'].value_counts().to_dict(),
    'job_distribution': df['job'].value_counts().to_dict(),
    'marital_distribution': df['marital'].value_counts().to_dict(),
    'education_distribution': df['education'].value_counts().to_dict(),
    'month_distribution': df['month'].value_counts().to_dict(),
    'age_stats': {'mean': float(df['age'].mean()), 'std': float(df['age'].std()),
                  'min': int(df['age'].min()), 'max': int(df['age'].max())},
    'balance_stats': {'mean': float(df['balance'].mean()), 'std': float(df['balance'].std()),
                      'min': int(df['balance'].min()), 'max': int(df['balance'].max())},
    'duration_stats': {'mean': float(df['duration'].mean()), 'std': float(df['duration'].std()),
                       'min': int(df['duration'].min()), 'max': int(df['duration'].max())},
    'campaign_stats': {'mean': float(df['campaign'].mean()), 'std': float(df['campaign'].std()),
                       'min': int(df['campaign'].min()), 'max': int(df['campaign'].max())},
    'y_by_job': df.groupby('job')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_education': df.groupby('education')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_marital': df.groupby('marital')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_month': df.groupby('month')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_age_group': df.groupby('age_group', observed=False)['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_balance_group': df.groupby('balance_group', observed=False)['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_housing': df.groupby('housing')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_loan': df.groupby('loan')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_contact': df.groupby('contact')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
    'y_by_poutcome': df.groupby('poutcome')['y'].apply(lambda x: (x == 'yes').mean()).to_dict(),
}

with open(OUTPUT_DIR / 'preprocess_stats.json', 'w', encoding='utf-8') as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print("统计结果已保存到: preprocess_stats.json")

print("\n" + "=" * 60)
print("数据预处理完成！")
print("=" * 60)
