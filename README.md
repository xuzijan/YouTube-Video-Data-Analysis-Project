# YouTube Video Data Analysis Project

English | [中文](README_CN.md)

A comprehensive data science project analyzing 155,000+ YouTube videos across 20 years (2006-2025), featuring advanced feature engineering, multi-view embedding fusion, graph analysis, and reproducible modular pipelines.

**Project Status**: ✅ Version 1.0 Complete + ✅ Version 2.0 Complete (Advanced Pipeline)  
**Total Scope**: 50+ files | 3,500+ lines of code | 2.5 GB data | 20-year timeline

---

## Project Overview

This project provides **two complementary versions**:

### Version 1.0: Classical ML Pipeline (5-Stage Notebook Analysis)
- Feature engineering (9 → 59 dimensions)
- Exploratory Data Analysis (EDA)
- Machine learning classification and clustering
- Predictive modeling
- Interactive dashboards

### Version 2.0: Advanced Pipeline (Modular Python Framework)
- **Local sentence-transformer embeddings** (384D) for semantic understanding
- **Multi-view feature fusion** (443D total: 59 engineered + 384 embedding)
- **Graph construction** with cosine similarity (15,863 nodes, 317K edges)
- **Dual clustering**: KMeans (embedding space) + Louvain (graph space)
- **Temporal evolution analysis** across 20 years
- **Theory-grounded derivations** with full documentation

### Key Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Dataset** | 155,704 videos | 2006-2025 (20 years) |
| **Channels** | 15,863 unique | Global diversity |
| **Features (v1.0)** | 59 engineered | 6.5x expansion from 9 original |
| **Embeddings (v2.0)** | 384D + 59D | Fused 443-dimensional vectors |
| **Graph (v2.0)** | 317K edges | Top-K cosine similarity |
| **Clusters (v2.0)** | 10 + 17 | KMeans + Louvain communities |
| **Visualizations** | 20+ charts | High-resolution (300 DPI) |
| **Notebooks (v1.0)** | 5 complete | Full analysis pipeline |
| **Python modules (v2.0)** | 8 modular | Reproducible, theory-grounded |

---

## Project Structure

### Version 1.0: Core Notebooks (5 files)

```
phase1_youtube_year_analysis.ipynb      Data preprocessing and yearly analysis
phase2_feature_engineering.ipynb        Feature engineering (9 → 59 features)
phase3_eda_analysis.ipynb               Exploratory Data Analysis
phase4_clustering_prediction.ipynb      ML modeling (classification, clustering, prediction)
phase5_insights_dashboard.ipynb         Interactive dashboards
```

### Version 2.0: Advanced Pipeline Modules

```
advanced/
├── __init__.py
├── config.py                           Global configuration management
├── run_pipeline.py                     Main pipeline entry point
│
├── llm_embeddings.py                   Stage 1: Local sentence-transformer embeddings
├── feature_fusion.py                   Stage 2: Multi-view feature fusion (443D)
├── graph_construction.py               Stage 3: Channel similarity graph (cosine top-K)
├── community_detection.py              Stage 4: Dual clustering (KMeans + Louvain)
├── temporal_analysis.py                Stage 5: Year-wise temporal evolution
└── export_for_web.py                   Web export (reserved for v3.0)

resultphoto.py                          Advanced visualization generation script
```

### Data Files

```
youtube_video.csv                       Original dataset (~155K videos)
engineered_features_raw.csv             Raw engineered features (59D)
engineered_features_scaled.csv          Scaled engineered features (59D, normalized)
feature_engineering_metadata.json       Feature metadata

datasets_by_year/                       Year-wise raw data (2006-2025)
advanced_outputs/                       v2.0 pipeline outputs
visualization_results/                  v2.0 generated visualizations
```

### Documentation

```
README.md                               English documentation (this file)
README_CN.md                            Chinese documentation
ADVANCED_PIPELINE_2.0_PAPER_EN.tex     IEEE-style academic paper (v2.0)
FILE_INVENTORY.md                       Complete file and data inventory
PROJECT_COMPLETION_REPORT.txt           v1.0 completion summary
```

### Visualizations and Results

**Version 1.0** (EDA & ML):
- 11 high-resolution charts (300 DPI)
- Dashboard visualizations
- Classification & clustering analysis

**Version 2.0** (Advanced):
- Network graph visualization (channel similarity)
- t-SNE embedding projection (cluster separation)
- Temporal heatmap (yearly trends)
- Community profiles summary
- Top channels ranking per cluster

---

## Pipeline Stages

### Version 1.0: Classical Analysis (5 Phases)

| Phase | Input | Output | Key Methods |
|-------|-------|--------|-----------|
| 1: Preprocessing | 155K raw records | 20 yearly datasets | Data cleaning, validation |
| 2: Feature Engineering | 9 features | 59 engineered features | Statistical + domain engineering |
| 3: EDA | 59D features | 11 visualizations | Correlation, distribution analysis |
| 4: ML Modeling | 59D features | Models + rankings | Classification, clustering, regression |
| 5: Dashboards | Model results | Interactive viz | Seaborn + Matplotlib |

### Version 2.0: Advanced Pipeline (5 Modular Stages)

| Stage | Module | Input | Output | Key Innovation |
|-------|--------|-------|--------|-----------------|
| 1 | `llm_embeddings.py` | Video titles | 384D embeddings | Local sentence-transformers |
| 2 | `feature_fusion.py` | 59D + 384D | 443D fused vectors | Early fusion, channel aggregation |
| 3 | `graph_construction.py` | 443D channels | Cosine graph (317K edges) | Top-K neighbor filtering |
| 4 | `community_detection.py` | Graph + embeddings | Dual partitions | KMeans + Louvain modularity |
| 5 | `temporal_analysis.py` | Year + assignments | 107 temporal records | Year-wise aggregation (2006-2025) |

**Key Characteristics**:
- ✅ Local-first: No external APIs or heavy infrastructure
- ✅ Modular: Each stage independent and reusable
- ✅ Reproducible: Fixed random seeds, deterministic execution
- ✅ Theory-grounded: Mathematical derivations for all methods
- ✅ Documented: Full academic paper with formulas and references

---

## Technical Stack

### Version 1.0 Dependencies
- **Data**: pandas, numpy
- **ML**: scikit-learn, xgboost
- **Viz**: matplotlib, seaborn
- **IDE**: Jupyter Notebook, VS Code

### Version 2.0 Additional Dependencies
- **Embeddings**: sentence-transformers (all-MiniLM-L12-v2, 384D)
- **Graph**: networkx, python-louvain
- **Optimization**: GPU support via CUDA 11.8+ (optional)

### Environment Specifications
- **Python Version**: 3.10+ (3.13.7 tested on Windows)
- **Execution Time**: CPU ~200s, GPU (4070S) ~65s
- **Memory**: ~4 GB for embeddings + graph
- **OS**: Windows, Linux, macOS supported

---

## Advanced Features (v2.0)

### Multi-View Embedding Fusion

**Architecture**:
```
Video Title → Encoder (384D) → Average  
Metadata    → Encoder (384D) ──┘ → Fused Vector (384D)

Combined with Engineered Features (59D) → Final 443D Vector
```

**Methods**:
- Local sentence-transformers: all-MiniLM-L12-v2
- Early fusion at video level
- Channel aggregation via mean pooling
- Normalization and deterministic processing

### Graph Construction
- **Similarity Metric**: Cosine distance (scale-invariant)
- **Sparsity**: Top-K neighbors (K=20, threshold=0.4)
- **Graph Size**: 15,863 nodes, 317,118 edges
- **Symmetrization**: Union of mutual neighborhoods
- **Complexity**: O(N·K·log K) for efficient scaling

### Dual Clustering Perspectives

**KMeans (Embedding Space)**:
- Objective: Minimize within-cluster squared distances
- Clusters: 10 (optimal from elbow method)
- Initialization: k-means++ seeding
- Assumption: Spherical, well-separated clusters

**Louvain (Graph Space)**:
- Objective: Maximize modularity Q
- Communities: 17 (from greedy optimization)
- Gain criterion: Resolution γ=1.0
- Assumption: Relational cohesion, scale-free structure

### Temporal Evolution Analysis
- **Granularity**: Year-wise aggregation (2006-2025)
- **Metrics**: Video count, total views per cluster/year
- **Normalization**: Handles missing years with zero-filling
- **Interpretations**: Lifecycle trends, seasonal effects, growth patterns

---

## Installation and Usage

### Version 1.0: Quick Start

```bash
# 1. Clone repository
git clone https://github.com/xuzijan/YouTube-Video-Data-Analysis-Project.git
cd YouTube-Video-Data-Analysis-Project

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter

# 3. Run notebooks sequentially
jupyter notebook phase1_youtube_year_analysis.ipynb
# ... continue with phase2 through phase5
```

### Version 2.0: Advanced Pipeline

```bash
# 1. Install advanced dependencies
pip install sentence-transformers networkx python-louvain

# 2. Run complete pipeline
python -m advanced.run_pipeline

# 3. Generate visualizations
python resultphoto.py
```

**Pipeline Output Locations**:
- `advanced_outputs/` - Embeddings, features, clusters, graphs
- `visualization_results/` - Network, t-SNE, temporal heatmaps

### Loading Results for Analysis

```python
import numpy as np
import pandas as pd

# Load embeddings and features
embeddings = np.load('advanced_outputs/channel_fused_features.npy')
metadata = pd.read_csv('advanced_outputs/channel_fused_features_index.csv')

# Load clustering results
clusters = pd.read_csv('advanced_outputs/channel_clusters_communities.csv')
top_channels = pd.read_csv('advanced_outputs/top_channels_per_cluster.csv')

# Load graph and temporal data
graph_edges = pd.read_csv('advanced_outputs/channel_graph_edges.csv')
temporal_stats = pd.read_csv('advanced_outputs/cluster_temporal_stats.csv')
```

---

## Core Contributions

### Theoretical Foundations (v2.0)

1. **Text Embedding Module**: Local sentence-transformers with proper preprocessing
2. **Feature Fusion Strategy**: Early fusion derivation and channel aggregation via mean pooling
3. **Graph Construction**: Cosine similarity with top-K sparsification
4. **Dual Clustering Theory**:
   - KMeans objective: $J(\{C_k\},\{\mu_k\}) = \sum_{k=1}^{K} \sum_{x_i\in C_k} \|x_i - \mu_k\|^2$
   - Louvain modularity: $Q = \frac{1}{2m} \sum_{i,j} \left( A_{ij} - \frac{k_i k_j}{2m} \right) \delta(c_i, c_j)$
5. **Temporal Aggregation**: Year-wise normalization and lifecycle pattern extraction
6. **t-SNE Dimensionality Reduction**: Local neighborhood preservation for visualization

### Practical Innovations

- ✅ **Local-first approach**: No cloud dependencies, full reproducibility
- ✅ **Modular architecture**: Each stage independently testable and reusable
- ✅ **Deterministic execution**: Fixed seeds for consistency across runs
- ✅ **Theory-grounded**: All methods backed by mathematical derivations
- ✅ **Scale efficiency**: Handles 155K videos × 15K channels on consumer hardware
- ✅ **Windows native**: Tested on Windows PowerShell v5.1

---

## Key Findings

### Data Statistics
- **Total videos analyzed**: 155,704 across 15,863 channels
- **Time span**: 2006-2025 (20 years)
- **Total views**: ~202.6 billion
- **Graph structure**: 317,118 edges in channel similarity network
- **Average channel degree**: 40 neighbors

### Clustering Insights
- **KMeans clusters**: 10 distinct groups in embedding space
- **Louvain communities**: 17 communities in graph space
- **Cluster-community agreement**: Significant overlap in major blocks
- **Graph density**: 0.25% (sparse, indicating selective neighborhoods)

### Temporal Dynamics
- **Recent growth**: 2024-2025 clusters show exponential expansion
- **Legacy decline**: Early YouTube channels show stability
- **Lifecycle patterns**: Identifiable growth → plateau → decline cycles
- **Content evolution**: Semantic shifts detected via embeddings

---

## Project Status & Roadmap

### ✅ Completed (v1.0 + v2.0)
- Full YouTube data analysis pipeline (155K videos, 20 years)
- Feature engineering (9 → 59 dimensions)
- Classical ML modeling (classification, clustering, regression)
- Advanced embedding fusion (443-dimensional vectors)
- Dual clustering perspectives (embeddings + graph)
- Temporal evolution analysis
- IEEE-style academic paper with full derivations

### 📋 Planned (v3.0)
- [ ] Web dashboard (Flask/FastAPI)
- [ ] Real-time update pipeline
- [ ] Multi-language support
- [ ] Recommendation system integration
- [ ] Causal inference on content success

---

## Citation

If you use this project in research, please cite:

```bibtex
@article{YouTube2025,
  title={Advanced Pipeline 2.0 for YouTube Ecosystem Analysis: 
         Multi-View Embedding Fusion, Graph Structure, and Temporal Dynamics},
  author={Analysis Team},
  year={2025},
  note={Local, Reproducible, and Theory-Grounded}
}
```

---

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Contact

For questions or collaboration inquiries, please contact the project maintainer.

---

**Project Statistics**:
- 📊 50+ files | 🧮 3,500+ lines of code | 💾 2.5 GB data
- ⚡ Execution time: 65s (GPU) or 200s (CPU)
- 📈 20-year dataset | 🌐 155K+ videos
- 🎯 100% reproducible with fixed seeds
