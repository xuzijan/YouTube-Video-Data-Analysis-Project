"""
YouTube 爬虫脚本1.0 - 使用 Service Account
"""
import sys
import os
import json
import argparse
from pathlib import Path
from datetime import datetime

from google.oauth2 import service_account
from googleapiclient.discovery import build


def create_youtube_service(credentials_path):
    credentials = service_account.Credentials.from_service_account_file(
        credentials_path,
        scopes=['https://www.googleapis.com/auth/youtube.readonly']
    )
    youtube = build('youtube', 'v3', credentials=credentials)
    return youtube


def search_videos(youtube_service, query, max_results=50):
    results = []
    next_page_token = None
    
    while len(results) < max_results:
        request = youtube_service.search().list(
            q=query,
            part='snippet',
            type='video',
            maxResults=min(50, max_results - len(results)),
            pageToken=next_page_token,
            order='viewCount'
        )
        
        response = request.execute()
        items = response.get('items', [])
        results.extend(items)
        
        next_page_token = response.get('nextPageToken')
        if not next_page_token or len(results) >= max_results:
            break
    
    return results[:max_results]


def main():
    parser = argparse.ArgumentParser(description="YouTube 简单爬虫")
    parser.add_argument("--query", type=str, default="machine learning",
                        help="搜索关键词")
    parser.add_argument("--count", type=int, default=50,
                        help="爬取数量")
    parser.add_argument("--output", type=str, default=None,
                        help="输出文件名")
    parser.add_argument("--client-secret", type=str, default="credentials.json",
                        help="Service Account 凭证文件")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.client_secret):
        print(f"找不到凭证文件: {args.client_secret}")
        sys.exit(1)
    
    try:
        print(f"初始化 YouTube 服务...")
        youtube = create_youtube_service(args.client_secret)
        
        print(f"搜索: '{args.query}' (最多 {args.count} 个)")
        results = search_videos(youtube, args.query, args.count)
        
        print(f"找到 {len(results)} 个视频\n")
        
        # 保存为 JSON
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"results/crawled_{timestamp}.json"
        else:
            output_file = args.output
        
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        file_size = os.path.getsize(output_file) / 1024
        print(f"已保存到: {output_file} ({file_size:.1f} KB)")
        
        # 显示前几条
        print("\n【数据样本】")
        for i, item in enumerate(results[:3]):
            snippet = item.get('snippet', {})
            print(f"\n{i+1}. {snippet.get('title', 'N/A')}")
            print(f"   频道: {snippet.get('channelTitle', 'N/A')}")
            print(f"   发布时间: {snippet.get('publishedAt', 'N/A')}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
