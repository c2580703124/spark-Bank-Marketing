# Spark 银行营销数据分析

基于 **Spark** 的银行营销数据集（Bank Marketing）分析项目，完成了 **数据预处理 → SparkSQL 多维分析 → SparkMLlib 建模 → 可视化呈现** 的完整数据分析流程，目标是根据客户属性与历史营销记录，**预测其是否会认购定期存款**。

## 项目简介

- **数据集**：Bank Marketing，4521 条记录 × 17 个特征
- **任务类型**：二分类（`y`：是否认购定期存款），训练集 / 测试集 = **7 : 3**
- **核心结论**：随机森林 AUC 达到 **0.8996**，`duration`（通话时长）、`month`（月份）、`job`（职业）是最关键的特征

| 模型 | AUC |
| --- | --- |
| **Random Forest** | **0.8996** |
| Logistic Regression | 0.8813 |
| Decision Tree | 0.5574 |

## 技术栈

| 环节 | 语言 / 框架 |
| --- | --- |
| 数据预处理 | Python（pandas、numpy） |
| 分布式分析与建模 | **Scala** + Spark 3.0.3（SparkSQL、SparkMLlib Pipeline） |
| 结果可视化 | Python（matplotlib） |

> 按实验要求，Spark 相关代码**全部使用 Scala 实现**，未使用 PySpark。

## 目录结构

```
spark-Bank-Marketing/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── data/                       # 数据集与字段说明
│   ├── bank.csv                # 原始数据（4521 × 17）
│   └── bank-names.txt          # 数据集字段说明
├── src/                        # 源码
│   ├── preprocess.py           # 数据预处理、特征工程
│   ├── spark_analysis.scala    # SparkSQL 多维分析 + SparkMLlib 建模
│   └── visualize.py            # 可视化
├── output/                     # 分析产物
│   ├── bank_processed.csv      # 预处理后的数据
│   ├── preprocess_stats.json   # 预处理统计结果
│   └── model_results.json      # 模型评估结果
├── plots/                      # 可视化图表（9 张）
└── docs/                       # 文档
    ├── 实验报告.md              # 完整实验报告
    ├── 实验要求与评分标准.md
    └── 实验总结与心得.md
```

## 快速开始

**环境要求**：JDK 1.8、Spark 3.0.3（Scala 2.12）、Python 3.x

```bash
# 1. 安装 Python 依赖
pip install -r requirements.txt

# 2. 数据预处理 —— 生成 output/bank_processed.csv、output/preprocess_stats.json
python src/preprocess.py

# 3. Spark 分析与建模 —— 生成 output/model_results.json
#    需在仓库根目录执行，脚本内数据路径为 data/bank.csv
spark-shell -i src/spark_analysis.scala

# 4. 生成可视化图表到 plots/
python src/visualize.py
```

## 核心分析结果

### 1. 目标变量分布 —— 数据明显不平衡

| y（是否认购） | 数量 | 占比 |
| --- | --- | --- |
| no | 4000 | 88.48% |
| yes | 521 | 11.52% |

![目标分布](plots/1_target_distribution.png)

### 2. 按月营销效果 —— 差异极其显著

10 月（46.25%）、12 月（45.00%）、3 月（42.86%）的订阅率远高于其他月份，5 月最低（6.65%）。

![按月营销效果](plots/3_monthly_effect.png)

### 3. 按职业的订阅率

退休人员（23.48%）与学生（22.62%）的订阅意愿最强，约为整体均值（11.52%）的 2 倍。

![按职业的订阅率](plots/2_job_subscription.png)

### 4. 模型性能对比

![模型性能对比](plots/4_model_comparison.png)

### 5. 综合仪表盘

![综合仪表盘](plots/comprehensive_dashboard.png)

> 更多分析维度（教育水平、婚姻状况、上次营销结果、联系方式、年龄段、余额档位）以及完整的运行结果说明，请见 **[docs/实验报告.md](docs/实验报告.md)**。

## 主要发现与营销建议

| 发现 | 建议 |
| --- | --- |
| 上次营销结果为 `success` 的客户，再次订阅率高达 **64.34%** | 优先对历史营销成功的客户做二次跟进 |
| 10 / 12 / 3 月订阅率 > 42%，5 月仅 6.65% | 将营销资源集中在这些高转化月份 |
| 退休人员、学生订阅意愿最强 | 重点面向退休与学生客群 |
| 通话时长是首要预测特征（重要性约 34%） | 优化话术、延长有效通话时长 |
| 联系方式为 telephone / cellular 时转化率约 14.5%，`unknown` 仅 4.61% | 保证联系方式有效性 |
| 高余额客户（2000–10000）订阅率 17.11% | 结合资产分层设计推荐策略 |

## 许可

本项目采用 [MIT License](LICENSE)。
