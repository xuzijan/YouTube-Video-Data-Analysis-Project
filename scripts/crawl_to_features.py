import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime
import logging

import numpy as np
import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def normalize_crawl_data(json_data):
    
 
    if isinstance(json_data, str):
        with open(json_data, 'r', encoding='utf-8') as f:
            items = json.load(f)
    else:
        items = json_data
    
    logger.info(f"规范化")
    
    rows = []
    for item in items:
        try:
            snippet = item.get('snippet', {})
            
            video_id = item.get('id', {}).get('videoId') or item.get('videoId')
            
            row = {
                'video_id': video_id,
                'title': snippet.get('title', 'N/A'),
                'channel_name': snippet.get('channelTitle', 'N/A'),
                'channel_id': snippet.get('channelId', 'N/A'),
                'view_count': 0,  
                'like_count': 0,
                'comment_count': 0,
                'published_date': snippet.get('publishedAt', ''),
                'thumbnail': snippet.get('thumbnails', {}).get('default', {}).get('url', '')
            }
            
            rows.append(row)
        except Exception as e:
            logger.warning(f"跳过一条记录: {e}")
            continue
    
    df = pd.DataFrame(rows)
    
    df['published_date'] = pd.to_datetime(df['published_date'], errors='coerce')
    df['view_count'] = pd.to_numeric(df['view_count'], errors='coerce').fillna(0).astype(int)
    df['like_count'] = pd.to_numeric(df['like_count'], errors='coerce').fillna(0).astype(int)
    df['comment_count'] = pd.to_numeric(df['comment_count'], errors='coerce').fillna(0).astype(int)
    
    logger.info(f"{len(df)} 条有效记录")
    return df

def engineer_text_features(df):
    df_features = df.copy()

    
    # 1. 标题长度
    df_features['title_length'] = df_features['title'].str.len()
    
    # 2. 标题词数
    df_features['title_word_count'] = df_features['title'].str.split().str.len()
    
    # 3. 平均词长
    df_features['title_avg_word_length'] = df_features.apply(
        lambda row: np.mean([len(word) for word in str(row['title']).split()]) 
                    if len(str(row['title']).split()) > 0 else 0,
        axis=1
    )
    
    # 4. 是否包含问号
    df_features['title_has_question'] = df_features['title'].str.contains(r'\?', regex=True).astype(int)
    
    # 5. 是否包含感叹号
    df_features['title_has_exclamation'] = df_features['title'].str.contains(r'!', regex=True).astype(int)
    
    # 6. 大写字母占比
    df_features['title_uppercase_ratio'] = df_features['title'].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / len(str(x)) if len(str(x)) > 0 else 0
    )
    
    # 7. 数字个数
    df_features['title_digit_count'] = df_features['title'].str.count(r'\d')
    
    # 8. 特殊字符个数
    df_features['title_special_char_count'] = df_features['title'].apply(
        lambda x: sum(1 for c in str(x) if not c.isalnum() and c not in ' ')
    )
    
    # 9. 是否包含数字
    df_features['title_has_numbers'] = df_features['title'].str.contains(r'\d', regex=True).astype(int)
    
    # 10. 是否包含括号
    df_features['title_has_brackets'] = df_features['title'].str.contains(r'[\(\[\{]', regex=True).astype(int)
    
    return df_features


def engineer_temporal_features(df):
    df_features = df.copy()
    ref_date = df_features['published_date'].max()
    
    # 1-4. 发布年月日时
    df_features['publish_year'] = df_features['published_date'].dt.year
    df_features['publish_month'] = df_features['published_date'].dt.month
    df_features['publish_day'] = df_features['published_date'].dt.day
    df_features['publish_hour'] = df_features['published_date'].dt.hour
    
    # 5. 一周中的第几天
    df_features['publish_dayofweek'] = df_features['published_date'].dt.dayofweek
    
    # 6. 是否周末
    df_features['is_weekend'] = (df_features['publish_dayofweek'] >= 5).astype(int)
    
    # 7. 是否工作时间
    df_features['is_working_hours'] = (
        (df_features['publish_hour'] >= 9) & 
        (df_features['publish_hour'] <= 17) & 
        (df_features['publish_dayofweek'] < 5)
    ).astype(int)
    
    # 8. 是否黄金时间 (19:00-22:00)
    df_features['is_prime_time'] = (
        (df_features['publish_hour'] >= 19) & 
        (df_features['publish_hour'] <= 22)
    ).astype(int)
    
    # 9-11. 距发布时间
    df_features['days_since_publish'] = (ref_date - df_features['published_date']).dt.days
    df_features['months_since_publish'] = df_features['days_since_publish'] // 30
    df_features['years_since_publish'] = df_features['days_since_publish'] // 365
    
    # 12. 季度
    df_features['publish_quarter'] = df_features['published_date'].dt.quarter
    
    # 13. 是否假期季节 (11-12月, 1月)
    df_features['is_holiday_season'] = (
        (df_features['publish_month'] >= 11) | 
        (df_features['publish_month'] <= 1)
    ).astype(int)
    
    return df_features

def engineer_engagement_features(df):
    df_features = df.copy()
    df_features['view_count'] = df_features['view_count'].fillna(1).clip(lower=1)
    df_features['like_count'] = df_features['like_count'].fillna(0)
    df_features['comment_count'] = df_features['comment_count'].fillna(0)
    
    # 1. 互动率
    df_features['engagement_rate'] = (
        (df_features['like_count'] + df_features['comment_count']) / 
        (df_features['view_count'] + 1)
    )
    
    # 2. 点赞率
    df_features['like_rate'] = df_features['like_count'] / (df_features['view_count'] + 1)
    
    # 3. 评论率
    df_features['comment_rate'] = df_features['comment_count'] / (df_features['view_count'] + 1)
    
    # 4. 点赞评论比
    df_features['like_to_comment_ratio'] = (
        df_features['like_count'] / (df_features['comment_count'] + 1)
    )
    
    # 5. 总互动数
    df_features['total_interactions'] = (
        df_features['like_count'] + df_features['comment_count']
    )
    
    # 6. 加权互动度 
    df_features['weighted_engagement'] = (
        df_features['like_count'] * 0.3 + 
        df_features['comment_count'] * 0.7
    ) / (df_features['view_count'] + 1)
    
    # 7. 互动多样性 
    total_interactions = df_features['total_interactions'].clip(lower=1)
    like_ratio = df_features['like_count'] / total_interactions
    comment_ratio = df_features['comment_count'] / total_interactions
    
    df_features['engagement_diversity'] = np.where(
        total_interactions > 0,
        -((like_ratio * np.log(like_ratio + 1e-10)) +
          (comment_ratio * np.log(comment_ratio + 1e-10))),
        0
    )
    
    # 8-11. 对数变换
    df_features['log_view_count'] = np.log1p(df_features['view_count'])
    df_features['log_like_count'] = np.log1p(df_features['like_count'])
    df_features['log_comment_count'] = np.log1p(df_features['comment_count'])
    df_features['log_engagement'] = np.log1p(df_features['total_interactions'])
    
    # 12-14. 病毒判断
    engagement_mean = df_features['engagement_rate'].mean()
    engagement_std = df_features['engagement_rate'].std()
    
    df_features['is_viral'] = (
        df_features['engagement_rate'] > (engagement_mean + engagement_std)
    ).astype(int)
    
    df_features['is_ultra_viral'] = (
        df_features['engagement_rate'] > (engagement_mean + 3 * engagement_std)
    ).astype(int)
    
    df_features['is_unpopular'] = (
        df_features['engagement_rate'] < (engagement_mean - engagement_std)
    ).astype(int)
    
    return df_features

def engineer_channel_features(df):
    df_features = df.copy()
    
    # 计算频道级统计
    channel_stats = df_features.groupby('channel_id').agg({
        'view_count': ['mean', 'sum', 'count', 'std'],
        'like_count': 'mean',
        'comment_count': 'mean',
        'engagement_rate': 'mean'
    }).reset_index()
    
    channel_stats.columns = [
        'channel_id', 'channel_avg_views', 'channel_total_views', 
        'channel_video_count', 'channel_view_std',
        'channel_avg_likes', 'channel_avg_comments', 'channel_avg_engagement'
    ]
    df_features = df_features.merge(channel_stats, on='channel_id', how='left')
    
    # 1-3. 已有的频道统计
    
    # 4. 频道一致性 (标准差/均值)
    df_features['channel_consistency_ratio'] = (
        df_features['channel_view_std'] / (df_features['channel_avg_views'] + 1)
    )
    
    # 5. 视频相对频道平均观看
    df_features['views_vs_channel_avg'] = (
        df_features['view_count'] / (df_features['channel_avg_views'] + 1)
    )
    
    # 6. 互动相对频道平均
    df_features['engagement_vs_channel_avg'] = (
        df_features['engagement_rate'] / (df_features['channel_avg_engagement'] + 1e-10)
    )
    
    # 7. 频道等级 (按视频数)
    def get_channel_tier(video_count):
        if video_count <= 10:
            return 1
        elif video_count <= 100:
            return 2
        elif video_count <= 1000:
            return 3
        else:
            return 4
    
    df_features['channel_tier'] = df_features['channel_video_count'].apply(get_channel_tier)
    
    # 8. 频道互动率
    df_features['channel_engagement_rate'] = df_features['channel_avg_engagement']
    
    # 9. 视频在频道内排名
    df_features['video_rank_in_channel'] = df_features.groupby('channel_id')['view_count'].rank(ascending=False)
    
    return df_features

def engineer_composite_features(df):
    df_features = df.copy()
    
    # 1. 标题长度-互动交互
    df_features['title_length_engagement_factor'] = (
        df_features['title_length'] * df_features['engagement_rate']
    )
    
    # 2. 长标题病毒视频
    df_features['long_title_viral'] = (
        (df_features['title_length'] > df_features['title_length'].median()) & 
        (df_features['is_viral'] == 1)
    ).astype(int)
    
    # 3. 内容新鲜度
    df_features['freshness_score'] = (
        (1.0 / (df_features['days_since_publish'] + 1)) * 100
    )
    
    # 4. 内容成熟度
    df_features['content_maturity_score'] = (
        (df_features['days_since_publish'] / 365) * df_features['engagement_rate'] * 1000
    )
    
    # 5. 频道适配度
    df_features['channel_fit_score'] = (
        df_features['views_vs_channel_avg'] * df_features['engagement_vs_channel_avg']
    )
    
    # 6. 小频道突破
    df_features['small_channel_breakout'] = (
        (df_features['channel_tier'] <= 2) & 
        (df_features['is_viral'] == 1)
    ).astype(int)
    
    # 7. 内容质量评分
    max_like = df_features['log_like_count'].max() + 1
    max_words = df_features['title_word_count'].max() + 1
    consistency = df_features['channel_consistency_ratio'].fillna(1).clip(0, 1)
    
    df_features['content_quality_score'] = (
        (df_features['engagement_rate'] * 0.3 +
         df_features['log_like_count'] / max_like * 0.3 +
         df_features['title_word_count'] / max_words * 0.2 +
         (1 - consistency) * 0.2) * 100
    )
    
    # 8. 发布时机评分
    df_features['publish_timing_score'] = (
        df_features['is_prime_time'] * 0.5 +
        (df_features['publish_dayofweek'] >= 3).astype(int) * 0.3 +
        df_features['is_weekend'] * 0.2
    ) * 100
    
    return df_features

def process_crawl_to_features(df_raw):
    df_features = engineer_text_features(df_raw)
    df_features = engineer_temporal_features(df_features)
    df_features = engineer_engagement_features(df_features)
    df_features = engineer_channel_features(df_features)
    df_features = engineer_composite_features(df_features)
    
    logger.info("=" * 60)
    
    # 统计
    original_cols = ['video_id', 'title', 'channel_name', 'channel_id', 
                     'view_count', 'like_count', 'comment_count', 'published_date', 'thumbnail']
    engineered_cols = [col for col in df_features.columns if col not in original_cols]
    
    logger.info(f"特征工程完成!")
    logger.info(f"   原始列: {len(original_cols)}")
    logger.info(f"   工程特征: {len(engineered_cols)}")
    logger.info(f"   总列数: {len(df_features.columns)}")
    logger.info(f"   数据行数: {len(df_features):,}")

    text_feat = [col for col in engineered_cols if 'title' in col]
    time_feat = [col for col in engineered_cols if any(x in col for x in 
                 ['publish_', 'is_', 'since', 'holiday', 'weekend', 'working', 'prime', 'days_', 'months_', 'years_', 'quarter'])]
    engagement_feat = [col for col in engineered_cols if any(x in col for x in 
                       ['engagement', 'rate', 'like_to', 'viral', 'unpopular', 'log_', 'interaction', 'weighted', 'diversity'])]
    channel_feat = [col for col in engineered_cols if 'channel' in col or 'rank' in col]
    composite_feat = [col for col in engineered_cols if any(x in col for x in 
                      ['length_engagement', 'long_title', 'freshness', 'maturity', 'fit', 'breakout', 'quality', 'timing'])]
    
    logger.info(f"\n特征类别分布:")
    logger.info(f"   文本特征: {len(text_feat)}")
    logger.info(f"   时间特征: {len(time_feat)}")
    logger.info(f"   互动特征: {len(engagement_feat)}")
    logger.info(f"   频道特征: {len(channel_feat)}")
    logger.info(f"   复合特征: {len(composite_feat)}")
    logger.info(f"   合计: {len(text_feat) + len(time_feat) + len(engagement_feat) + len(channel_feat) + len(composite_feat)}")
    
    return df_features

def main():
    parser = argparse.ArgumentParser(description="爬虫输出")
    
    # 输入方式选择
    parser.add_argument("--input", type=str, default=None,
                        help="输入 JSON 文件路径")
    parser.add_argument("--crawl", action='store_true',
                        help="直接爬取数据（跳过 --input）")
    parser.add_argument("--query", type=str, default="machine learning",
                        help="爬取关键词（与 --crawl 配合）")
    parser.add_argument("--count", type=int, default=50,
                        help="爬取数量（与 --crawl 配合）")
    
    # 输出配置
    parser.add_argument("--output", type=str, default=None,
                        help="输出 CSV 文件名")
    parser.add_argument("--client-secret", type=str, default="credentials.json",
                        help="Google Service Account 凭证文件")
    
    args = parser.parse_args()
    
    try:
        if args.crawl:
            logger.info(f"启动爬虫: 查询='{args.query}', 数量={args.count}")
            
            if not os.path.exists(args.client_secret):
                logger.error(f"找不到凭证文件: {args.client_secret}")
                sys.exit(1)
            
            credentials = service_account.Credentials.from_service_account_file(
                args.client_secret,
                scopes=['https://www.googleapis.com/auth/youtube.readonly']
            )
            youtube = build('youtube', 'v3', credentials=credentials)
        
            results = []
            next_page_token = None
            
            while len(results) < args.count:
                request = youtube.search().list(
                    q=args.query,
                    part='snippet',
                    type='video',
                    maxResults=min(50, args.count - len(results)),
                    pageToken=next_page_token,
                    order='viewCount'
                )
                
                response = request.execute()
                items = response.get('items', [])
                results.extend(items)
                
                next_page_token = response.get('nextPageToken')
                if not next_page_token or len(results) >= args.count:
                    break
            
            results = results[:args.count]
            df_raw = normalize_crawl_data(results)
            
        else:
            if not args.input:
                logger.error("必须指定 --input 或 --crawl")
                sys.exit(1)
            
            if not os.path.exists(args.input):
                logger.error(f"找不到输入文件: {args.input}")
                sys.exit(1)
            
            logger.info(f"读取输入文件: {args.input}")
            df_raw = normalize_crawl_data(args.input)
        
        df_features = process_crawl_to_features(df_raw)
        
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/features_{timestamp}.csv"
        else:
            output_file = args.output
        
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        df_features.to_csv(output_file, index=False, encoding='utf-8')
        file_size = os.path.getsize(output_file) / 1024
        
        logger.info(f"已保存到: {output_file} ({file_size:.1f} KB)")
        logger.info(f"   数据形状: {df_features.shape[0]} 行 × {df_features.shape[1]} 列")
        
        # 显示前几列
        logger.info(f"\n【数据预览】")
        logger.info(f"\n原始列:")
        logger.info(df_features[['video_id', 'title', 'channel_name']].head(2).to_string())
        logger.info(f"\n特征样本:")
        feature_cols = ['title_length', 'engagement_rate', 'is_viral', 'channel_tier', 'freshness_score']
        logger.info(df_features[feature_cols].head(2).to_string())
        
        return 0
        
    except Exception as e:
        logger.error(f"错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
