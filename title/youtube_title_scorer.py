"""
YouTube标题质量打分器
基于梯度提升 + DeepSeek API的创造力评估
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import re
import warnings
from datetime import datetime
import requests
import json
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# ==================== 配置 ====================
# 从config.py导入配置
try:
    from config import DEEPSEEK_API_KEY, DEEPSEEK_API_URL, DATA_PATH
except ImportError:
    # 如果config.py不存在，使用默认值
    DEEPSEEK_API_KEY = "your_api_key_here"
    DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
    DATA_PATH = "youtube_video.csv"

# ==================== 数据加载与初步探索 ====================

def load_data(filepath: str) -> pd.DataFrame:
    """加载YouTube视频数据"""
    df = pd.read_csv(filepath)
    print(f"数据加载完成: {len(df)} 条记录")
    print(f"列信息: {df.columns.tolist()}")
    return df

def calculate_engagement_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算互动指标"""
    df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce')
    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce')
    df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce')
    
    # 互动率（处理0值）
    df['engagement_rate'] = (df['like_count'] + df['comment_count']) / (df['view_count'] + 1)
    df['like_rate'] = df['like_count'] / (df['view_count'] + 1)
    df['comment_rate'] = df['comment_count'] / (df['view_count'] + 1)
    
    # 对数变换（缓解异常值）
    df['log_engagement'] = np.log1p(df['engagement_rate'] * 1000)
    
    return df

# ==================== 文本特征工程 ====================

def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """从标题提取文本特征"""
    
    features = {
        'title_length': [],
        'title_word_count': [],
        'uppercase_ratio': [],
        'digit_count': [],
        'exclamation_count': [],
        'question_count': [],
        'parenthesis_count': [],
        'pipe_count': [],
        'dash_count': [],
        'emoji_count': [],
        'has_colon': [],
        'has_brand': [],
        'all_caps_words': [],
    }
    
    for title in df['title']:
        if pd.isna(title):
            title = ""
        
        # 基础长度
        features['title_length'].append(len(title))
        features['title_word_count'].append(len(title.split()))
        
        # 大写比例
        upper_chars = sum(1 for c in title if c.isupper())
        features['uppercase_ratio'].append(upper_chars / (len(title) + 1))
        
        # 数字
        features['digit_count'].append(len(re.findall(r'\d', title)))
        
        # 特殊字符统计
        features['exclamation_count'].append(title.count('!'))
        features['question_count'].append(title.count('?'))
        features['parenthesis_count'].append(title.count('(') + title.count(')'))
        features['pipe_count'].append(title.count('|'))
        features['dash_count'].append(title.count('-'))
        
        # Emoji统计（简单版本）
        emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE)
        features['emoji_count'].append(len(emoji_pattern.findall(title)))
        
        # 冒号和品牌标记
        features['has_colon'].append(1 if ':' in title else 0)
        features['has_brand'].append(1 if any(x in title.lower() for x in ['official', '#shorts', 'trailer']) else 0)
        
        # 全大写词数量
        all_caps = len([w for w in title.split() if w.isupper() and len(w) > 1])
        features['all_caps_words'].append(all_caps)
    
    feature_df = pd.DataFrame(features)
    print(f"文本特征提取完成: {len(feature_df.columns)} 个特征")
    
    return pd.concat([df, feature_df], axis=1)

# ==================== DeepSeek API 调用 ====================

def call_deepseek_api(title: str, aspect: str = "creativity") -> float:
    """
    调用DeepSeek API评估标题的创造力/新颖性等主观维度
    
    Args:
        title: 视频标题
        aspect: 评估维度 ('creativity' / 'emotional_appeal' / 'clarity')
    
    Returns:
        评分 0-10
    """
    
    prompts = {
        "creativity": f"""请评估以下YouTube视频标题的创造力和新颖性，返回0-10的分数，其中10表示极具创意：
标题: "{title}"
只返回一个数字，不要其他文字。""",
        
        "emotional_appeal": f"""请评估以下YouTube视频标题的情感吸引力和好奇心激发程度，返回0-10的分数：
标题: "{title}"
只返回一个数字，不要其他文字。""",
        
        "clarity": f"""请评估以下YouTube视频标题的清晰度和易理解程度，返回0-10的分数：
标题: "{title}"
只返回一个数字，不要其他文字。"""
    }
    
    prompt = prompts.get(aspect, prompts["creativity"])
    
    try:
        response = requests.post(
            DEEPSEEK_API_URL,
            headers={
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 10
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            score_text = result['choices'][0]['message']['content'].strip()
            # 提取数字
            score = float(re.findall(r'\d+\.?\d*', score_text)[0])
            return min(10, max(0, score))  # 限制在0-10
        else:
            print(f"API错误: {response.status_code}")
            return 5.0  # 默认值
    except Exception as e:
        print(f"API调用失败: {e}")
        return 5.0

def batch_deepseek_scoring(titles: List[str], aspect: str = "creativity", 
                           sample_size: int = None, delay: float = 0.5) -> Dict[str, float]:
    """
    批量评估标题（带采样和延迟控制）
    
    Args:
        titles: 标题列表
        aspect: 评估维度
        sample_size: 采样数量（不指定则全部评估）
        delay: 请求间隔秒数
    
    Returns:
        标题->评分字典
    """
    
    if sample_size:
        indices = np.random.choice(len(titles), min(sample_size, len(titles)), replace=False)
        sampled_titles = [titles[i] for i in indices]
    else:
        sampled_titles = titles
    
    scores = {}
    print(f"\n开始DeepSeek API评估 ({len(sampled_titles)}条)...")
    
    for i, title in enumerate(sampled_titles):
        if pd.isna(title):
            scores[title] = 5.0
        else:
            score = call_deepseek_api(title, aspect)
            scores[title] = score
            
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{len(sampled_titles)}")
            
            time.sleep(delay)
    
    print(f"DeepSeek评估完成")
    return scores

def add_deepseek_features(df: pd.DataFrame, sample_size: int = 1000) -> pd.DataFrame:
    """
    为数据集添加DeepSeek评分特征
    
    注意：需要配置有效的API密钥
    """
    
    print(f"\nDeepSeek API集成模式 (采样{sample_size}条进行评估)")
    print(f"   API密钥状态: {'✓ 已配置' if DEEPSEEK_API_KEY != 'your_api_key_here' else '✗ 未配置'}")
    
    if DEEPSEEK_API_KEY == "your_api_key_here":
        print(" 跳过API调用，使用模拟数据...")
        # 模拟数据（用于演示）
        np.random.seed(42)
        df['creativity_score'] = np.random.uniform(4, 9, len(df))
        df['emotional_appeal_score'] = np.random.uniform(3, 9, len(df))
        df['clarity_score'] = np.random.uniform(5, 10, len(df))
    else:
        # 真实API调用
        creativity_scores = batch_deepseek_scoring(
            df['title'].tolist(), 
            aspect="creativity", 
            sample_size=sample_size
        )
        
        emotional_scores = batch_deepseek_scoring(
            df['title'].tolist(), 
            aspect="emotional_appeal", 
            sample_size=sample_size
        )
        
        df['creativity_score'] = df['title'].map(
            lambda x: creativity_scores.get(x, 5.0)
        )
        df['emotional_appeal_score'] = df['title'].map(
            lambda x: emotional_scores.get(x, 5.0)
        )
        df['clarity_score'] = np.random.uniform(5, 10, len(df))  # 备用
    
    print(f" DeepSeek特征添加完成")
    return df

# ==================== 模型训练 ====================

def prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """准备特征和目标变量"""
    
    # 选择所有数值特征（排除ID和原始计数和目标变量）
    exclude_cols = {
        'video_id', 'title', 'channel_name', 'channel_id', 
        'thumbnail', 'published_date', 'view_count', 
        'like_count', 'comment_count', 'engagement_rate',
        'like_rate', 'comment_rate', 'log_engagement'  # 添加log_engagement到排除列表
    }
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # 移除包含NaN的行
    df_clean = df[feature_cols + ['log_engagement']].dropna()
    
    X = df_clean[feature_cols]
    y = df_clean['log_engagement']
    
    print(f"\n特征准备完成:")
    print(f"   样本数: {len(df_clean)}")
    print(f"   特征数: {len(feature_cols)}")
    print(f"   特征列表: {feature_cols}")
    
    return X, y, feature_cols

def train_model(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Tuple[xgb.XGBRegressor, np.ndarray, np.ndarray]:
    """训练XGBoost模型"""
    
    # 数据分割
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 训练XGBoost
    print("\n训练XGBoost模型...")
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )
    
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        early_stopping_rounds=10,
        verbose=False
    )
    
    # 评估
    y_pred = model.predict(X_test_scaled)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"模型训练完成:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   RMSE: {rmse:.4f}")
    
    return model, scaler, X_test_scaled, y_test, X_test

# ==================== 特征重要性分析 ====================

def analyze_feature_importance(model: xgb.XGBRegressor, X_test: pd.DataFrame, 
                               y_test: pd.Series, feature_cols: List[str]):
    """使用SHAP分析特征贡献度"""
    
    print("\nSHAP特征重要性分析...")
    
    # 计算SHAP值（使用采样以加快速度）
    try:
        explainer = shap.TreeExplainer(model)
        # 只使用前1000个样本以避免内存问题
        sample_size = min(1000, len(X_test))
        X_sample = X_test.iloc[:sample_size].copy()
        X_sample.columns = range(len(X_sample.columns))  # 重命名列为数字，避免重复
        shap_values = explainer.shap_values(X_sample)
        
        # 绘制特征重要性
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        plt.title("SHAP特征重要性排名\n（对互动率预测的平均影响）", fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('feature_importance_shap.png', dpi=300, bbox_inches='tight')
        print(f"特征重要性图表已保存: feature_importance_shap.png")
        plt.close()
        
        # 绘制SHAP力图（示例）
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        plt.title("SHAP摘要图", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('shap_summary_plot.png', dpi=300, bbox_inches='tight')
        print(f"SHAP摘要已保存: shap_summary_plot.png")
        plt.close()
    except Exception as e:
        print(f"SHAP分析失败: {str(e)[:100]}")
        shap_values = None
        explainer = None
    
    return shap_values, explainer

# ==================== 实时打分工具 ====================

def score_single_title(title: str, model: xgb.XGBRegressor, 
                       scaler, feature_cols: List[str]) -> Dict:
    """为单个标题打分"""
    
    # 提取特征
    df_single = pd.DataFrame({'title': [title]})
    df_single = extract_text_features(df_single)
    
    # 模拟DeepSeek评分（实际应调用API）
    df_single['creativity_score'] = np.random.uniform(4, 9, 1)
    df_single['emotional_appeal_score'] = np.random.uniform(3, 9, 1)
    df_single['clarity_score'] = np.random.uniform(5, 10, 1)
    
    # 准备特征（使用相同的feature_cols）
    X_single = df_single[feature_cols]
    X_single_scaled = scaler.transform(X_single)
    
    # 预测
    log_engagement_pred = model.predict(X_single_scaled)[0]
    engagement_rate_pred = (np.exp(log_engagement_pred) - 1) / 1000
    
    # 提取特征值用于解释
    feature_values = X_single.iloc[0].to_dict()
    
    return {
        'title': title,
        'predicted_engagement_rate': max(0, engagement_rate_pred),
        'predicted_log_engagement': log_engagement_pred,
        'feature_values': feature_values,
        'creativity_score': df_single['creativity_score'].values[0],
        'emotional_appeal': df_single['emotional_appeal_score'].values[0]
    }

# ==================== 主函数 ====================

def main():
    """主流程"""
    
    print("=" * 60)
    print("YouTube标题质量打分器 v1.0")
    print("=" * 60)
    
    # 1. 加载数据
    df = load_data(DATA_PATH)
    
    # 2. 计算互动指标
    df = calculate_engagement_metrics(df)
    print(f"\n互动指标统计:")
    print(f"   平均互动率: {df['engagement_rate'].mean():.6f}")
    print(f"   中位数: {df['engagement_rate'].median():.6f}")
    
    # 3. 提取文本特征
    df = extract_text_features(df)
    
    # 4. 添加DeepSeek评分
    df = add_deepseek_features(df, sample_size=500)
    
    # 5. 准备特征
    X, y, feature_cols = prepare_features(df)
    
    # 6. 训练模型
    model, scaler, X_test_scaled, y_test, X_test = train_model(X, y)
    
    # 7. 特征重要性分析
    shap_values, explainer = analyze_feature_importance(model, X_test, y_test, feature_cols)
    
    # 8. 示例打分
    print("\n" + "=" * 60)
    print("实时打分示例")
    print("=" * 60)
    
    test_titles = [
        "Why do Human Feet Wash up on This Beach? | Fascinating Horror Shorts",
        "The ULTIMATE iPhone 15 Review - You NEED to Watch This!",
        "Cooking Tutorial #256",
        "🔥 SHOCKING Truth About AI That They Don't Want You to Know!!!",
    ]
    
    for title in test_titles:
        result = score_single_title(title, model, scaler, feature_cols)
        print(f"\n 标题: {title[:50]}...")
        print(f"   预测互动率: {result['predicted_engagement_rate']:.4f}")
        print(f"   创造力评分: {result['creativity_score']:.1f}/10")
        print(f"   情感吸引力: {result['emotional_appeal']:.1f}/10")
    
    print("\n" + "=" * 60)
    print("  生成的图表:")
    print("      - feature_importance_shap.png")
    print("      - shap_summary_plot.png")
    print("=" * 60)
    
    return model, scaler, feature_cols, df

if __name__ == "__main__":
    model, scaler, feature_cols, df = main()
