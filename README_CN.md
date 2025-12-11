# YouTube 视频数据分析项目

[English](README.md) | 中文

一个全面的数据科学项目，分析了155,000多个YouTube视频，时间跨度20年（2006-2025），包含高级特征工程、多视角嵌入融合、图分析和可复现的模块化管道。

**项目状态**: ✅ 版本 1.0 完成 + ✅ 版本 2.0 完成（高级管道）  
**完整范围**: 50+ 文件 | 3,500+ 行代码 | 2.5 GB 数据 | 20 年时间跨度

---

## 项目概览

本项目提供**两个互补的版本**：

### 版本 1.0：经典 ML 管道（5 阶段笔记本分析）
- 特征工程（9 → 59 维）
- 探索性数据分析（EDA）
- 机器学习分类和聚类
- 预测建模
- 交互式仪表板

### 版本 2.0：高级管道（模块化 Python 框架）
- **本地句子级转换器嵌入**（384D）用于语义理解
- **多视角特征融合**（443D 总和：59 工程特征 + 384 嵌入）
- **图构建**采用余弦相似性（15,863 个节点，317K 条边）
- **双聚类**：KMeans（嵌入空间）+ Louvain（图空间）
- **20 年时间演化分析**
- **理论依据的推导**和完整文档

### 关键指标

| 指标 | 数值 | 说明 |
|------|------|------|
| **数据集** | 155,704 条视频 | 2006-2025（20 年） |
| **频道数** | 15,863 个独特 | 全球多样性 |
| **特征 (v1.0)** | 59 个工程特征 | 从 9 个原始特征扩展 6.5 倍 |
| **嵌入 (v2.0)** | 384D + 59D | 融合 443 维向量 |
| **图 (v2.0)** | 317K 条边 | 顶 K 余弦相似性 |
| **聚类 (v2.0)** | 10 + 17 | KMeans + Louvain 社区 |
| **可视化** | 20+ 图表 | 高分辨率（300 DPI） |
| **笔记本 (v1.0)** | 5 个完整 | 完整分析管道 |
| **Python 模块 (v2.0)** | 8 个模块化 | 可复现，理论依据 |

---

## 项目结构

### 版本 1.0：核心笔记本（5 个文件）

```
phase1_youtube_year_analysis.ipynb      数据预处理和年度分析
phase2_feature_engineering.ipynb        特征工程（9 → 59 特征）
phase3_eda_analysis.ipynb               探索性数据分析
phase4_clustering_prediction.ipynb      ML 建模（分类、聚类、预测）
phase5_insights_dashboard.ipynb         交互式仪表板
```

### 版本 2.0：高级管道模块

```
advanced/
├── __init__.py
├── config.py                           全局配置管理
├── run_pipeline.py                     主管道入口
│
├── llm_embeddings.py                   阶段 1：本地句子级转换器嵌入
├── feature_fusion.py                   阶段 2：多视角特征融合（443D）
├── graph_construction.py               阶段 3：频道相似图（余弦顶K）
├── community_detection.py              阶段 4：双聚类（KMeans + Louvain）
├── temporal_analysis.py                阶段 5：逐年时间演化
└── export_for_web.py                   网页导出（v3.0 预留）

resultphoto.py                          高级可视化生成脚本
```

### 数据文件

```
youtube_video.csv                       原始数据集（~155K 视频）
engineered_features_raw.csv             原始工程特征（59D）
engineered_features_scaled.csv          标准化工程特征（59D，已归一化）
feature_engineering_metadata.json       特征元数据

datasets_by_year/                       按年份原始数据（2006-2025）
advanced_outputs/                       v2.0 管道输出
visualization_results/                  v2.0 生成的可视化
```

### 文档

```
README.md                               英文文档
README_CN.md                            中文文档（本文件）
ADVANCED_PIPELINE_2.0_PAPER_EN.tex     IEEE 风格学术论文（v2.0）
FILE_INVENTORY.md                       完整文件和数据清单
PROJECT_COMPLETION_REPORT.txt           v1.0 完成总结
```

### 可视化和结果

**版本 1.0**（EDA 和 ML）：
- 11 个高分辨率图表（300 DPI）
- 仪表板可视化
- 分类与聚类分析

**版本 2.0**（高级）：
- 网络图可视化（频道相似度）
- t-SNE 嵌入投影（聚类分离）
- 时间热力图（年度趋势）
- 社区档案总结
- 每个聚类的顶级频道排名

---

## 管道阶段

### 版本 1.0：经典分析（5 个阶段）

| 阶段 | 输入 | 输出 | 关键方法 |
|------|------|------|--------|
| 1：预处理 | 155K 原始记录 | 20 个按年数据集 | 数据清洗、验证 |
| 2：特征工程 | 9 个特征 | 59 个工程特征 | 统计 + 域特征 |
| 3：EDA | 59D 特征 | 11 个可视化 | 相关性、分布分析 |
| 4：ML 建模 | 59D 特征 | 模型 + 排名 | 分类、聚类、回归 |
| 5：仪表板 | 模型结果 | 交互式可视化 | Seaborn + Matplotlib |

### 版本 2.0：高级管道（5 个模块阶段）

| 阶段 | 模块 | 输入 | 输出 | 创新点 |
|------|------|------|------|--------|
| 1 | `llm_embeddings.py` | 视频标题 | 384D 嵌入 | 本地句子级转换器 |
| 2 | `feature_fusion.py` | 59D + 384D | 443D 融合向量 | 早期融合，频道聚合 |
| 3 | `graph_construction.py` | 443D 频道 | 余弦图（317K 边） | 顶 K 邻域过滤 |
| 4 | `community_detection.py` | 图 + 嵌入 | 双分割 | KMeans + Louvain 模块性 |
| 5 | `temporal_analysis.py` | 年份 + 分配 | 107 个时间记录 | 逐年聚合（2006-2025） |

**关键特性**：
- ✅ 本地优先：无需外部 API 或重型基础设施
- ✅ 模块化：每个阶段独立且可重用
- ✅ 可复现：固定随机种子，确定性执行
- ✅ 理论依据：所有方法都有数学推导支持
- ✅ 完整文档：包含公式和参考的学术论文

---

## 技术栈

### 版本 1.0 依赖
- **数据处理**：pandas、numpy
- **机器学习**：scikit-learn、xgboost
- **可视化**：matplotlib、seaborn
- **IDE**：Jupyter Notebook、VS Code

### 版本 2.0 额外依赖
- **嵌入**：sentence-transformers（all-MiniLM-L12-v2，384D）
- **图**：networkx、python-louvain
- **优化**：CUDA 11.8+ GPU 支持（可选）

### 环境要求
- **Python 版本**：3.10+（Windows 上测试 3.13.7）
- **执行时间**：CPU ~200 秒，GPU (4070S) ~65 秒
- **内存需求**：~4 GB（嵌入 + 图）
- **操作系统**：Windows、Linux、macOS 均支持

---

## 高级特性（v2.0）

### 多视角嵌入融合

**架构**：
```
视频标题 → 编码器（384D） → 平均
元数据  → 编码器（384D） ──┘ → 融合向量（384D）

与工程特征（59D）结合 → 最终 443D 向量
```

**方法**：
- 本地句子级转换器：all-MiniLM-L12-v2
- 视频级早期融合
- 频道级聚合（平均池化）
- 归一化和确定性处理

### 图构建
- **相似性度量**：余弦距离（尺度不变）
- **稀疏性**：顶 K 邻域（K=20，阈值=0.4）
- **图大小**：15,863 个节点，317,118 条边
- **对称化**：相互邻域的并集
- **复杂度**：高效缩放的 O(N·K·log K)

### 双聚类视角

**KMeans（嵌入空间）**：
- 目标：最小化类内平方距离
- 聚类数：10（从肘部法则确定）
- 初始化：k-means++ 种子
- 假设：球形、良好分离的聚类

**Louvain（图空间）**：
- 目标：最大化模块性 Q
- 社区数：17（从贪心优化得出）
- 增益准则：分辨率 γ=1.0
- 假设：关系凝聚力，无标度结构

### 时间演化分析
- **粒度**：逐年聚合（2006-2025）
- **指标**：每个聚类/年的视频数、总观看数
- **归一化**：缺失年份用零填充
- **解释**：生命周期趋势、季节效应、增长模式

---

## 安装和使用

### 版本 1.0：快速开始

```bash
# 1. 克隆仓库
git clone https://github.com/xuzijan/YouTube-Video-Data-Analysis-Project.git
cd YouTube-Video-Data-Analysis-Project

# 2. 安装依赖
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# 3. 按顺序运行笔记本
jupyter notebook phase1_youtube_year_analysis.ipynb
# ... 继续运行 phase2 到 phase5
```

### 版本 2.0：高级管道

```bash
# 1. 安装高级依赖
pip install sentence-transformers networkx python-louvain

# 2. 运行完整管道
python -m advanced.run_pipeline

# 3. 生成可视化
python resultphoto.py
```

**管道输出位置**：
- `advanced_outputs/` - 嵌入、特征、聚类、图
- `visualization_results/` - 网络图、t-SNE、时间热力图

### 加载结果进行分析

```python
import numpy as np
import pandas as pd

# 加载嵌入和特征
embeddings = np.load('advanced_outputs/channel_fused_features.npy')
metadata = pd.read_csv('advanced_outputs/channel_fused_features_index.csv')

# 加载聚类结果
clusters = pd.read_csv('advanced_outputs/channel_clusters_communities.csv')
top_channels = pd.read_csv('advanced_outputs/top_channels_per_cluster.csv')

# 加载图和时间数据
graph_edges = pd.read_csv('advanced_outputs/channel_graph_edges.csv')
temporal_stats = pd.read_csv('advanced_outputs/cluster_temporal_stats.csv')
```

---

## 核心贡献

### 理论基础（v2.0）

1. **文本嵌入模块**：带有适当预处理的本地句子级转换器
2. **特征融合策略**：早期融合推导和通过平均池化的频道聚合
3. **图构建**：采用顶 K 稀疏化的余弦相似性
4. **双聚类理论**：
   - KMeans 目标：$J(\{C_k\},\{\mu_k\}) = \sum_{k=1}^{K} \sum_{x_i\in C_k} \|x_i - \mu_k\|^2$
   - Louvain 模块性：$Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$
5. **时间聚合**：逐年归一化和生命周期模式提取
6. **t-SNE 降维**：用于可视化的局部邻域保留

### 实践创新

- ✅ **本地优先方法**：无云依赖，完全可复现
- ✅ **模块化架构**：每个阶段独立可测试和可重用
- ✅ **确定性执行**：固定种子保证跨运行一致性
- ✅ **理论依据**：所有方法都有数学推导支持
- ✅ **规模效率**：在消费级硬件上处理 155K 视频 × 15K 频道
- ✅ **Windows 原生**：在 Windows PowerShell v5.1 上测试

---

## 关键发现

### 数据统计
- **分析视频总数**：155,704 个，跨越 15,863 个频道
- **时间跨度**：2006-2025（20 年）
- **总观看数**：~202.6 十亿
- **图结构**：频道相似网络中 317,118 条边
- **平均频道度数**：40 个邻域

### 聚类洞察
- **KMeans 聚类**：嵌入空间中 10 个不同群组
- **Louvain 社区**：图空间中 17 个社区
- **聚类-社区一致性**：主块中显著重叠
- **图密度**：0.25%（稀疏，表示选择性邻域）

### 时间动态
- **近期增长**：2024-2025 聚类显示指数扩展
- **遗留下降**：早期 YouTube 频道显示稳定性
- **生命周期模式**：可识别的增长 → 平台 → 下降循环
- **内容演化**：通过嵌入检测到的语义转变

---

## 项目状态与路线图

### ✅ 已完成（v1.0 + v2.0）
- 完整 YouTube 数据分析管道（155K 视频，20 年）
- 特征工程（9 → 59 维）
- 经典 ML 建模（分类、聚类、回归）
- 高级嵌入融合（443 维向量）
- 双聚类视角（嵌入 + 图）
- 时间演化分析
- IEEE 风格学术论文，包含完整推导

### 📋 计划（v3.0）
- [ ] 网页仪表板（Flask/FastAPI）
- [ ] 实时更新管道
- [ ] 多语言支持
- [ ] 推荐系统集成
- [ ] 内容成功因果推断

---

## 引用

如果在研究中使用本项目，请按以下方式引用：

```bibtex
@article{YouTube2025,
  title={Advanced Pipeline 2.0 for YouTube Ecosystem Analysis: 
         Multi-View Embedding Fusion, Graph Structure, and Temporal Dynamics},
  author={分析团队},
  year={2025},
  note={本地、可复现且理论依据}
}
```

---

## 许可证

MIT 许可证 - 详见 LICENSE 文件

## 贡献

欢迎贡献！请随时提交问题或拉取请求。

## 联系方式

如有问题或合作意向，请联系项目维护者。

---

**项目统计**：
- 📊 50+ 文件 | 🧮 3,500+ 行代码 | 💾 2.5 GB 数据
- ⚡ 执行时间：65 秒（GPU）或 200 秒（CPU）
- 📈 20 年数据集 | 🌐 155K+ 视频
- 🎯 100% 可复现（固定种子）

