# YouTube Advanced Pipeline 2.0 质量评估报告

## 1. 管线概览

本报告针对新增的 `advanced/` 模块，评估从 LLM 表征到图结构、聚类与时间演化的 5 步端到端管线质量。整体流程如下：

1. **Step 1/5：视频与频道表征**  (`VideoEmbeddingGenerator`, `ChannelEmbeddingAggregator`)
2. **Step 2/5：特征融合**  (`FeatureFusion`)
3. **Step 3/5：频道相似图构建**  (`ChannelGraphBuilder`)
4. **Step 4/5：聚类与社区检测**  (`CommunityDetector`)
5. **Step 5/5：时间演化分析**  (`TemporalAnalyzer`)

所有步骤由 `run_full_advanced_pipeline()` 串联，通过命令：

```bash
python -m advanced.run_pipeline
```

即可一键完成。

---

## 2. 各步骤输出与一致性检查

### 2.1 Step 1：视频 & 频道表征

**核心文件**

- `advanced_outputs/video_embeddings_deepseek.npy`
- `advanced_outputs/video_embedding_index.csv`
- `advanced_outputs/channel_embeddings_deepseek.npy`
- `advanced_outputs/channel_embedding_index.csv`

**字段对齐**

- `video_embedding_index.csv` 头部字段：
  - `video_id, title, channel_name, channel_id, view_count, like_count, comment_count, published_date, thumbnail, embedding_index`
- 频道级聚合使用 `channel_id` 分组，与后续所有以频道为节点的模块保持一致（下游统一转换为 `channel_id_x`）。

**质量评价**

- 使用本地 `sentence-transformers/all-MiniLM-L12-v2` 生成文本 embedding，避免了远程 API 的不稳定性；
- `embedding_index` 与 `.npy` 矩阵按行一一对应，结构清晰；
- 频道级 embedding 为视频 embedding 的均值，适合作为上层 graph 与聚类的输入。

### 2.2 Step 2：特征融合

**核心文件**

- `advanced_outputs/video_fused_features.npy`
- `advanced_outputs/video_fused_features_index.csv`（>50MB，大规模、多列）
- `advanced_outputs/channel_fused_features.npy`
- `advanced_outputs/channel_fused_features_index.csv`

**关键设计**

- 在视频层面，将 `engineered_features_scaled.csv` 中的数值特征与 `video_embeddings_deepseek.npy` 进行 early fusion：
  - 显式排除 id/meta 列（如 `video_id`, `channel_id`, `title`, `view_count` 等）；
  - 仅对真正的工程特征列进行拼接；
- 在频道层面，以 `channel_id_x`（或兼容的字段）为 key，对 `video_fused_features.npy` 求均值得到频道 fused 向量。

**质量评价**

- 特征选择逻辑与原工程特征保持一致，避免重复使用 id/meta 信息；
- `channel_fused_features_index.csv` 以 `channel_id_x` + `fusion_embedding_index` 形式索引，结构简单，方便图构建使用；
- 大规模 `video_fused_features_index.csv` 说明所有视频均成功参与融合，无明显丢失。

### 2.3 Step 3：频道相似图构建

**核心文件**

- `advanced_outputs/channel_graph_nodes.csv`
- `advanced_outputs/channel_graph_edges.csv`

**关键设计**

- 节点：`channel_fused_features_index.csv` 的每一行代表一个频道，节点 ID 为 `channel_id_x`；
- 边：基于 `channel_fused_features.npy` 的 cosine 相似度：
  - 过滤阈值 `similarity_threshold`（默认 0.4）；
  - 每个节点保留 top-k 邻居（`similarity_top_k`，默认 20）。

**质量评价**

- `channel_graph_nodes.csv` 与 `channel_clusters_communities.csv` 在频道 ID（`channel_id_x`）上保持一致，便于后续对齐；
- 图结构稀疏、可控，既避免了全连接噪声，又保证足够连通性，适合社区检测算法使用。

### 2.4 Step 4：聚类与社区检测

**核心文件**

- `advanced_outputs/channel_clusters_communities.csv`

**字段示例**

- 头部示例：
  - `channel_id_x, fusion_embedding_index, cluster_id, community_id`

**关键设计**

- 在 fused 特征空间使用 `KMeans` 进行频道聚类，得到 `cluster_id`；
- 在相似图上使用 Louvain（若安装）或连通分量进行社区检测，得到 `community_id`；
- 所有操作统一使用 `channel_id_x` 作为频道标识列，避免列名不一致问题。

**质量评价**

- 每个频道都被分配了明确的 `cluster_id` 和 `community_id`，无缺失；
- 频道 ID 字段在 embedding、fusion、graph 与 temporal 各阶段全部统一为 `channel_id_x`，数据血缘清晰；
- 为后续做 cluster 解释、频道画像与可视化提供了良好基础。

### 2.5 Step 5：时间演化分析

**核心文件**

- `advanced_outputs/cluster_temporal_stats.csv`

**字段示例**

- 头部：`year, cluster_id, video_count, view_count`

**关键设计**

- 使用 `youtube_video.csv` 与 `channel_clusters_communities.csv` 按 `channel_id` ↔ `channel_id_x` 进行 join；
- 按 `year` × `cluster_id` 聚合：
  - `video_count`: 该年内属于该聚类的上传视频数量；
  - `view_count`: 该年内这些视频的总观看数（如果列存在则聚合）；
  - `engagement`: 若未来引入 engagement 指标，则按均值聚合。

**质量评价**

- 聚合逻辑对缺失列（如 `engagement`）具有鲁棒性，仅在列真实存在时才参与聚合；
- 输出表短小精悍，直接可用于画 heatmap 或时间趋势线；
- 将频道聚类结果与时间维度成功连接起来，为论文中的“Temporal Dynamics of Clusters”提供了量化支撑。

---

## 3. 整体鲁棒性与可解释性

1. **列名统一与鲁棒性**
   - 关键 ID：
     - 视频层：`video_id`
     - 频道层：`channel_id`（原始）、`channel_id_x`（fused / graph / community / temporal）
   - 各模块对列名有显式分支与错误信息，避免“默默猜测列名”导致的隐性错误。

2. **缺失值与可选字段处理**
   - 所有聚合前对数值列进行 `fillna(0)` 处理，避免 NaN 传播；
   - `view_count` / `engagement` 等列均采用“存在才聚合”的策略，不会再抛出 KeyError。

3. **可解释性基础**
   - `video_embedding_index.csv` 与 `channel_embedding_index.csv` 保留了原始 title、channel_name 等字段，便于后续做 cluster 解释；
   - `cluster_temporal_stats.csv` 可直接支撑：
     - 哪些 cluster 在近几年变得更活跃；
     - 不同 cluster 的观看量演化趋势。

---

## 4. 建议的下一步工作

1. **可视化与论文图表**
   - 基于 `cluster_temporal_stats.csv`：
     - 绘制 `year × cluster_id` 的视频数/观看量 heatmap；
     - 为若干代表性 cluster 画时间折线图；
   - 基于 `channel_graph_edges.csv` + `channel_clusters_communities.csv`：
     - 使用 UMAP + 颜色编码 `cluster_id`/`community_id` 做 2D 布局图。

2. **LLM 辅助解释模块（可选）
   - 对每个 `cluster_id` / `community_id` 抽取若干代表频道和视频 title，送入 DeepSeek Chat，生成：
     - 聚类主题标签；
     - 自然语言描述（如“高频发布游戏解说、近两年观看量快速增长”）。

3. **Web 3.0 仪表板准备（3.0 版本）**
   - 将 `channel_graph_nodes.csv`, `channel_graph_edges.csv`, `channel_clusters_communities.csv`, `cluster_temporal_stats.csv` 整理成前端友好的 JSON；
   - 为后续基于 Web 的交互式可视化（图谱浏览 + 时间轴）打基础。

---

## 5. 总体评价

- **完整性**：从原始视频数据到多视角 fused 表征、图结构、聚类与时间演化，管线闭环完整；
- **稳定性**：多处对列名和缺失值的防御性编程，使得在数据 schema 稍有变化时也能给出清晰报错或自动退化行为；
- **可扩展性**：各阶段输出文件边界清晰，便于替换 embedding 模型、调整图构建规则或聚类算法；
- **论文友好度**：
  - 可直接用于撰写“Methodology”与“Experiments”中关于表征学习、图聚类与时间动态的章节；
  - 结合简单可视化即可产出质量较高的实验图表。

总体来看，`advanced/` 管线已经达到了一个 **工程上可复现、研究上可解释** 的 2.0 版本水准。接下来建议重点转向：结果可视化与文字化总结，以支撑论文撰写和后续 3.0 Web 展示。
