# YouTube Video Data Analysis Project（readme writen by ai）

English | [中文](README_CN.md)

A comprehensive data science project analyzing 155,669 YouTube videos across 19 years (2006-2025), featuring advanced feature engineering, machine learning modeling, and interactive visualizations.

---

## Project Overview

This project applies data science methodologies to YouTube video data, including:
- Feature engineering (9 → 59 features)
- Exploratory Data Analysis (EDA)
- Machine learning classification and clustering
- Predictive modeling
- Interactive dashboards

### Key Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Data Size** | 155,669 records | 19-year historical data |
| **Features** | 59 engineered | 6.5x expansion from 9 original |
| **Classification** | High accuracy | Random Forest classifier |
| **Clustering** | Optimal K determined | K-means with silhouette analysis |
| **Visualizations** | 11 charts | High-resolution (300 DPI) |
| **Notebooks** | 5 complete | Full analysis pipeline |

---

## Project Structure

### Analysis Notebooks (5 Files)

```
phase1_youtube_year_analysis.ipynb      Data preprocessing and yearly analysis
phase2_feature_engineering.ipynb        Feature engineering (9 → 59 features)
phase3_eda_analysis.ipynb               Exploratory Data Analysis
phase4_clustering_prediction.ipynb      ML modeling (classification, clustering, prediction)
phase5_insights_dashboard.ipynb         Interactive dashboards
```

### Data Files

```
youtube_video.csv                       Original dataset (32 MB)
engineered_features_raw.csv             Raw engineered features (109 MB)
engineered_features_scaled.csv          Scaled features (207 MB)
feature_engineering_metadata.json       Feature metadata
datasets_by_year/                       20 yearly datasets (2006-2025)
```

### Visualizations (11 PNG Files in photo/)

```
Dashboard Visualizations (3 files):
  dashboard_overview.png                Overall data overview
  dashboard_classification.png          Classification model performance
  dashboard_clustering.png              Clustering analysis results

EDA Analysis Charts (5 files):
  top_videos_analysis.png               Top videos analysis
  engagement_analysis.png               Engagement rate analysis
  temporal_patterns.png                 Temporal patterns
  channel_comparison.png                Channel comparison
  feature_correlation_heatmap.png       Feature correlation heatmap
  feature_variance.png                  Feature variance analysis

ML Model Results (3 files):
  classification_analysis.png           Classification results
  channel_clustering_analysis.png       Channel clustering results
```

### Documentation

```
PROJECT_COMPLETION_REPORT.txt           Project completion summary
README.md                               This file
```

---

## Analysis Pipeline

### Phase 1: Data Preprocessing
- Input: 155,669 YouTube video records
- Output: 20 yearly datasets (2006-2025)
- Tasks: Data cleaning, validation, initial exploration

### Phase 2: Feature Engineering
- Input: 9 original features
- Output: 59 engineered features
  - Text features (12): Title length, word count, special characters, etc.
  - Temporal features (17): Year, month, day of week, hour, time slots, etc.
  - Engagement features (19): Engagement rate, like rate, comment rate, viral indicators, etc.
  - Channel features (15): Channel averages, totals, rankings, tiers, consistency, etc.
  - Composite features (8): Video vs. channel comparisons, quality scores, etc.

### Phase 3: Exploratory Data Analysis
- 6 analysis themes:
  - Top video rankings analysis
  - Engagement rate deep dive
  - Temporal patterns discovery
  - Channel comparison analysis
  - Video feature analysis
  - Comprehensive insights
- Output: 5 high-resolution visualizations

### Phase 4: Machine Learning Modeling
- Classification: Video engagement level prediction
  - Logistic Regression
  - Random Forest (best performance)
  - Support Vector Machine (SVM)
- Clustering: Channel type segmentation
  - K-means clustering
  - Elbow method + Silhouette analysis
  - Optimal K determination
- Prediction: Performance forecasting
  - Random Forest Regressor
  - XGBoost Regressor
- Feature importance analysis

### Phase 5: Interactive Dashboards
- 3 comprehensive dashboards:
  - Overall data overview
  - Classification model performance
  - Clustering analysis results

---

## Technical Stack

**Programming Language**: Python 3.x

**Data Processing**:
- pandas: Data manipulation
- numpy: Numerical computing

**Machine Learning**:
- scikit-learn: ML algorithms
- xgboost: Gradient boosting

**Visualization**:
- matplotlib: Static plots
- seaborn: Statistical visualizations

**Development Environment**:
- Jupyter Notebook
- VS Code

---

## Key Features

### Feature Engineering (59 Total Features)

**Text Features (12)**:
- Title length, word count
- Special character count and ratio
- Uppercase letter ratio
- Digit count
- Question mark presence
- Exclamation mark count

**Temporal Features (17)**:
- Publish year, month, day of week, hour
- Weekend indicator
- Prime time indicator (7-10 PM)
- Work hours indicator
- Holiday season indicator
- Days since publish

**Engagement Features (19)**:
- Engagement rate (main metric)
- Like rate, comment rate
- Weighted engagement
- Like-to-comment ratio
- Viral video indicator
- Ultra-viral indicator
- Unpopular video indicator

**Channel Features (15)**:
- Channel average views, likes, comments
- Channel total views
- Channel video count
- Channel tier classification
- Channel consistency ratio
- Channel engagement rate

**Composite Features (8)**:
- Title length × engagement interaction
- Freshness × quality score
- Views vs. channel average
- Content quality score
- Publishing timing score

### Machine Learning Models

**Classification Models**:
- Binary classification: High vs. Low engagement
- Models: Logistic Regression, Random Forest, SVM
- Evaluation: Accuracy, Precision, Recall, F1-Score
- Feature importance ranking

**Clustering Models**:
- K-means clustering on channels
- Optimal K selection using elbow method and silhouette coefficient
- Channel segmentation analysis

**Regression Models**:
- Target: Engagement rate prediction
- Models: Random Forest Regressor, XGBoost
- Evaluation: R², RMSE, MAE

---

## Installation and Usage

### Prerequisites

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

### Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Quick Start

1. Clone the repository
2. Ensure `youtube_video.csv` is in the project root directory
3. Run the notebooks in sequence:
   - `phase1_youtube_year_analysis.ipynb`
   - `phase2_feature_engineering.ipynb`
   - `phase3_eda_analysis.ipynb`
   - `phase4_clustering_prediction.ipynb`
   - `phase5_insights_dashboard.ipynb`

### File Structure

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
├── photo/                          (11 visualization files)
├── datasets_by_year/               (20 yearly datasets)
├── PROJECT_COMPLETION_REPORT.txt
└── README.md
```

---

## Key Findings

### Data Analysis Insights

1. **Top Video Characteristics**
   - View counts reach billions
   - Engagement rates show long-tail distribution
   - Viral videos represent small percentage

2. **Temporal Patterns**
   - Publishing time impact analysis
   - Day of week and seasonal effects
   - Optimal publishing windows identified

3. **Channel Analysis**
   - Significant scale differences between channels
   - Channel consistency vs. engagement correlation
   - Four-tier channel classification

4. **Engagement Deep Dive**
   - Overall engagement rate statistics
   - Viral video feature identification
   - Like-to-comment ratio patterns
   - View count segment analysis

### Machine Learning Results

**Classification Performance**:
- High accuracy achieved
- Random Forest identified as best model
- Top features identified through importance analysis

**Clustering Results**:
- Optimal cluster number determined
- Channel segments identified
- Distinct characteristics per cluster

**Prediction Models**:
- Strong predictive performance
- Feature importance ranking completed
- Key drivers of engagement identified

---

## Project Highlights

- **Large-scale data**: 155,669 records spanning 19 years
- **Comprehensive analysis**: 5-phase complete pipeline
- **Advanced feature engineering**: 6.5x feature expansion
- **Multiple ML algorithms**: Classification, clustering, regression
- **Professional visualizations**: 11 high-resolution charts (300 DPI)
- **Clean code**: English comments with Chinese function descriptions
- **Reproducible**: Complete Jupyter notebooks with clear structure

---

## Notes

- All code comments are in English for international standards
- Each notebook begins with a Chinese function description
- Visualizations are saved at 300 DPI for publication quality
- Feature engineering pipeline is modular and reusable

---
