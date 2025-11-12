# YouTube视频数据分析项目（readme由人工智能生成）

[English](README.md) | 中文

一个全面的数据科学项目，分析了155,669个YouTube视频，时间跨度19年（2006-2025），包含高级特征工程、机器学习建模和交互式可视化。

---

## 项目概述

本项目将数据科学方法应用于YouTube视频数据，包括：
- 特征工程（9个 → 59个特征）
- 探索性数据分析（EDA）
- 机器学习分类和聚类
- 预测建模
- 交互式仪表板

### 关键指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **数据规模** | 155,669条记录 | 19年历史数据 |
| **特征数量** | 59个工程特征 | 从9个原始特征扩展6.5倍 |
| **分类准确率** | 高准确率 | 随机森林分类器 |
| **聚类质量** | 最优K值确定 | K-means轮廓系数分析 |
| **可视化** | 11个图表 | 高分辨率（300 DPI） |
| **笔记本** | 5个完整 | 完整分析流程 |

---

## 项目结构

### 分析笔记本（5个文件）

```
phase1_youtube_year_analysis.ipynb      数据预处理和年度分析
phase2_feature_engineering.ipynb        特征工程（9 → 59个特征）
phase3_eda_analysis.ipynb               探索性数据分析
phase4_clustering_prediction.ipynb      机器学习建模（分类、聚类、预测）
phase5_insights_dashboard.ipynb         交互式仪表板
```

### 数据文件

```
youtube_video.csv                       原始数据集（32 MB）
engineered_features_raw.csv             原始工程特征（109 MB）
engineered_features_scaled.csv          标准化特征（207 MB）
feature_engineering_metadata.json       特征元数据
datasets_by_year/                       20个年度数据集（2006-2025）
```

### 可视化文件（photo/文件夹中的11个PNG文件）

```
仪表板可视化（3个文件）：
  dashboard_overview.png                整体数据概览
  dashboard_classification.png          分类模型性能
  dashboard_clustering.png              聚类分析结果

EDA分析图表（5个文件）：
  top_videos_analysis.png               顶级视频分析
  engagement_analysis.png               参与度分析
  temporal_patterns.png                 时间模式
  channel_comparison.png                频道对比
  feature_correlation_heatmap.png       特征相关性热力图
  feature_variance.png                  特征方差分析

机器学习模型结果（3个文件）：
  classification_analysis.png           分类结果
  channel_clustering_analysis.png       频道聚类结果
```

### 文档

```
PROJECT_COMPLETION_REPORT.txt           项目完成总结
README.md                               英文版说明文档
README_CN.md                            本文件（中文版）
```

---

## 分析流程

### 阶段1：数据预处理
- 输入：155,669条YouTube视频记录
- 输出：20个年度数据集（2006-2025）
- 任务：数据清洗、验证、初步探索

### 阶段2：特征工程
- 输入：9个原始特征
- 输出：59个工程特征
  - 文本特征（12个）：标题长度、词数、特殊字符等
  - 时间特征（17个）：年、月、星期、小时、时段等
  - 参与度特征（19个）：参与率、点赞率、评论率、病毒指标等
  - 频道特征（15个）：频道平均值、总计、排名、层级、一致性等
  - 复合特征（8个）：视频与频道对比、质量评分等

### 阶段3：探索性数据分析
- 6个分析主题：
  - 顶级视频排名分析
  - 参与度深度分析
  - 时间模式发现
  - 频道对比分析
  - 视频特征分析
  - 综合洞察
- 输出：5个高分辨率可视化

### 阶段4：机器学习建模
- 分类：视频参与度等级预测
  - 逻辑回归
  - 随机森林（最佳性能）
  - 支持向量机（SVM）
- 聚类：频道类型分割
  - K-means聚类
  - 肘部法则 + 轮廓系数分析
  - 最优K值确定
- 预测：性能预测
  - 随机森林回归器
  - XGBoost回归器
- 特征重要性分析

### 阶段5：交互式仪表板
- 3个综合仪表板：
  - 整体数据概览
  - 分类模型性能
  - 聚类分析结果

---

## 技术栈

**编程语言**：Python 3.x

**数据处理**：
- pandas：数据操作
- numpy：数值计算

**机器学习**：
- scikit-learn：机器学习算法
- xgboost：梯度提升

**可视化**：
- matplotlib：静态图表
- seaborn：统计可视化

**开发环境**：
- Jupyter Notebook
- VS Code

---

## 核心特征

### 特征工程（共59个特征）

**文本特征（12个）**：
- 标题长度、词数
- 特殊字符数量和比率
- 大写字母比率
- 数字计数
- 问号存在
- 感叹号数量

**时间特征（17个）**：
- 发布年、月、星期、小时
- 周末指标
- 黄金时段指标（晚7-10点）
- 工作时间指标
- 节假日季节指标
- 发布后天数

**参与度特征（19个）**：
- 参与率（主要指标）
- 点赞率、评论率
- 加权参与度
- 点赞评论比
- 病毒视频指标
- 超级病毒指标
- 不受欢迎视频指标

**频道特征（15个）**：
- 频道平均观看、点赞、评论
- 频道总观看数
- 频道视频数量
- 频道层级分类
- 频道一致性比率
- 频道参与率

**复合特征（8个）**：
- 标题长度×参与度交互
- 新鲜度×质量评分
- 观看数与频道平均值对比
- 内容质量评分
- 发布时机评分

### 机器学习模型

**分类模型**：
- 二元分类：高参与度 vs 低参与度
- 模型：逻辑回归、随机森林、SVM
- 评估：准确率、精确率、召回率、F1分数
- 特征重要性排名

**聚类模型**：
- 频道K-means聚类
- 使用肘部法则和轮廓系数选择最优K值
- 频道分割分析

**回归模型**：
- 目标：参与率预测
- 模型：随机森林回归器、XGBoost
- 评估：R²、RMSE、MAE

---

## 安装和使用

### 前置要求

```bash
Python 3.x
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
jupyter
```

### 安装

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### 快速开始

1. 克隆仓库
2. 确保 `youtube_video.csv` 在项目根目录
3. 按顺序运行笔记本：
   - `phase1_youtube_year_analysis.ipynb`
   - `phase2_feature_engineering.ipynb`
   - `phase3_eda_analysis.ipynb`
   - `phase4_clustering_prediction.ipynb`
   - `phase5_insights_dashboard.ipynb`

### 文件结构

```
5002/
├── phase1_youtube_year_analysis.ipynb
├── phase2_feature_engineering.ipynb
├── phase3_eda_analysis.ipynb
├── phase4_clustering_prediction.ipynb
├── phase5_insights_dashboard.ipynb
├── youtube_video.csv
├── engineered_features_raw.csv
├── engineered_features_scaled.csv
├── feature_engineering_metadata.json
├── photo/                          (11个可视化文件)
├── datasets_by_year/               (20个年度数据集)
├── PROJECT_COMPLETION_REPORT.txt
├── README.md                       (英文版)
└── README_CN.md                    (中文版)
```

---

## 关键发现

### 数据分析洞察

1. **顶级视频特征**
   - 观看数达到数十亿
   - 参与率呈现长尾分布
   - 病毒视频占比很小

2. **时间模式**
   - 发布时间影响分析
   - 星期和季节效应
   - 识别最佳发布窗口

3. **频道分析**
   - 频道间规模差异显著
   - 频道一致性与参与度相关
   - 四层频道分类

4. **参与度深度分析**
   - 整体参与率统计
   - 识别病毒视频特征
   - 点赞评论比模式
   - 观看数分段分析

### 机器学习结果

**分类性能**：
- 实现高准确率
- 随机森林为最佳模型
- 通过重要性分析识别关键特征

**聚类结果**：
- 确定最优聚类数
- 识别频道分段
- 每个聚类的独特特征

**预测模型**：
- 强大的预测性能
- 完成特征重要性排名
- 识别参与度关键驱动因素

---

## 项目亮点

- **大规模数据**：155,669条记录，跨越19年
- **全面分析**：5阶段完整流程
- **高级特征工程**：特征扩展6.5倍
- **多种机器学习算法**：分类、聚类、回归
- **专业可视化**：11个高分辨率图表（300 DPI）
- **清晰代码**：英文注释配中文功能说明
- **可重现**：完整的Jupyter笔记本，结构清晰

---

## 说明

- 所有代码注释均为英文，符合国际标准
- 每个笔记本开头均有中文功能描述
- 可视化以300 DPI保存，适合出版质量
- 特征工程流程模块化且可重用

---

