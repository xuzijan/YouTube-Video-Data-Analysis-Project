"""
YouTube标题打分工具
可直接输入标题，获得评分和改进建议
"""

import json
import pickle
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

class TitleScorer:
    """标题打分器"""
    
    def __init__(self, model_path='model.pkl', scaler_path='scaler.pkl'):
        """初始化打分器"""
        self.model = None
        self.scaler = None
        self.feature_cols = [
            'title_length', 'word_count', 'uppercase_ratio', 'digit_count',
            'exclamation', 'question', 'emoji_count', 'has_colon', 'has_pipe',
            'punctuation_density', 'creativity_score', 'emotional_appeal', 'novelty_score'
        ]
        
        # 加载模型（如果存在）
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            self.load_model(model_path, scaler_path)
    
    def load_model(self, model_path, scaler_path):
        """加载预训练模型"""
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print("模型加载成功")
        except Exception as e:
            print(f"模型加载失败: {e}")
    
    def save_model(self, model, scaler, model_path='model.pkl', scaler_path='scaler.pkl'):
        """保存模型"""
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"模型已保存: {model_path}, {scaler_path}")
    
    def extract_features(self, title):
        """从标题提取特征"""
        
        if pd.isna(title) or not title:
            title = ""
        
        features = {}
        
    
        features['title_length'] = len(title)
        features['word_count'] = len(title.split())
    
        upper_chars = sum(1 for c in title if c.isupper())
        features['uppercase_ratio'] = upper_chars / (len(title) + 1)
      
        features['digit_count'] = len(re.findall(r'\d', title))
        
     
        features['exclamation'] = title.count('!')
        features['question'] = title.count('?')

        emoji_pattern = re.compile("["
            "\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
            "\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF"
            "\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
            "\U00002702-\U000027B0\U000024C2-\U0001F251" "]+", flags=re.UNICODE)
        features['emoji_count'] = len(emoji_pattern.findall(title))
        

        features['has_colon'] = 1 if ':' in title else 0
        features['has_pipe'] = 1 if '|' in title else 0
        

        punct_count = len(re.findall(r'[!?.,;:\-|()[\]{}]', title))
        features['punctuation_density'] = punct_count / (len(title.split()) + 1)
        
        # AI评分（启发式）
        creativity = 5 + len(title) / 30 - features['uppercase_ratio'] * 2 + features['emoji_count'] * 1.5
        features['creativity_score'] = min(10, max(3, creativity))
        
        emotional = 5 + features['question'] * 1.5 + features['exclamation'] * 1.2 + features['emoji_count'] * 0.8
        features['emotional_appeal'] = min(10, max(2, emotional))
        
        unique_words = len(set(title.lower().split()))
        novelty = 5 + (unique_words / len(title.split())) * 3 if features['word_count'] > 0 else 5
        features['novelty_score'] = min(10, max(3, novelty))
        
        return features
    
    def score(self, title):
        """为标题打分"""
        
        features = self.extract_features(title)
        
    
        result = {
            'title': title,
            'features': features,
            'creativity_score': features['creativity_score'],
            'emotional_appeal': features['emotional_appeal'],
            'novelty_score': features['novelty_score'],
        }
        
        # 如果有完整模型，添加互动率预测
        if self.model and self.scaler:
            try:
                X_single = pd.DataFrame([features])[self.feature_cols]
                X_single_scaled = self.scaler.transform(X_single)
                log_engagement = self.model.predict(X_single_scaled)[0]
                engagement_rate = (np.exp(log_engagement) - 1) / 1000
                result['predicted_engagement_rate'] = max(0, engagement_rate)
            except Exception as e:
                print(f"预测失败: {e}")
        
        return result
    
    # def generate_suggestions(self, title, score_result):
    #     """改进建议"""
        
    #     suggestions = []
    #     features = score_result['features']
        
    #     # 长度建议
    #     if features['title_length'] < 30:
    #         suggestions.append("标题过短，建议增加到40-70字符以提供更多信息")
    #     elif features['title_length'] > 100:
    #         suggestions.append("标题过长，易被截断，建议控制在80字符以内")
    #     else:
    #         suggestions.append("✓ 标题长度适中")
        
    #     # 标点符号建议
    #     if features['question'] == 0:
    #         suggestions.append("考虑添加问号(?)以激发好奇心 (预期+8%互动)")
    #     elif features['question'] >= 2:
    #         suggestions.append("问号过多，可能显得生硬，建议1-2个")
        
    #     if features['exclamation'] >= 3:
    #         suggestions.append("感叹号过多，可能显得浮夸")
        
    #     # Emoji建议
    #     if features['emoji_count'] == 0:
    #         suggestions.append("可考虑添加1-2个相关Emoji (预期+12%互动)")
    #     elif features['emoji_count'] > 3:
    #         suggestions.append("Emoji太多，显得混乱，建议1-2个")
        
    #     # 创造力建议
    #     if score_result['creativity_score'] < 5:
    #         suggestions.append("创意度较低，考虑使用更新颖的词汇或视角")
        
    #     # 情感吸引力
    #     if score_result['emotional_appeal'] < 5:
    #         suggestions.append("情感吸引力不足，可增加悬念或紧迫感词汇")
        
    #     # 新颖度
    #     if score_result['novelty_score'] < 5:
    #         suggestions.append("新颖度有限，考虑避免常见的标题模式")
        
    #     return suggestions
    
    def compare_titles(self, titles_list):
        """对比多个标题"""
        
        print("\n" + "="*80)
        print("标题对比分析")
        print("="*80)
        
        results = []
        for i, title in enumerate(titles_list, 1):
            score = self.score(title)
            results.append(score)
            
            engagement = score.get('predicted_engagement_rate', 0)
            print(f"\n#{i} 标题: {title}")
            print(f"    创造力: {score['creativity_score']:.1f}/10")
            print(f"    情感吸引: {score['emotional_appeal']:.1f}/10")
            print(f"    新颖度: {score['novelty_score']:.1f}/10")
            if engagement > 0:
                print(f"    预期互动率: {engagement:.4f} ({engagement*100:.2f}%)")
        
        # 排名
        print("\n" + "-"*80)
        print("排名（按预期互动率）:")
        ranked = sorted(
            [(i, r.get('predicted_engagement_rate', 0)) for i, r in enumerate(results, 1)],
            key=lambda x: x[1],
            reverse=True
        )
        for rank, (idx, engagement) in enumerate(ranked, 1):
            print(f"  {rank}. 标题#{idx}: {engagement*100:.2f}%")
        
        return results
    
    def interactive_mode(self):
        """交互式打分"""
        
        print("\n" + "="*80)
        print("YouTube标题打分工具 - 交互模式")
        print("="*80)
        print("\n命令:")
        print("  score <title>   - 打分单个标题")
        print("  compare         - 对比多个标题")
        print("  quit            - 退出")
        print("-"*80)
        
        while True:
            user_input = input("\n输入命令: ").strip()
            
            if user_input.lower() == 'quit':
                print("再见！")
                break
            
            if user_input.lower().startswith('score '):
                title = user_input[6:].strip()
                if title:
                    result = self.score(title)
                    self.print_result(result)
                    suggestions = self.generate_suggestions(title, result)
                    print("\n改进建议:")
                    for s in suggestions:
                        print(f"  {s}")
            
            elif user_input.lower() == 'compare':
                titles = []
                print("输入标题(空行结束):")
                while True:
                    t = input("  > ").strip()
                    if not t:
                        break
                    titles.append(t)
                if titles:
                    self.compare_titles(titles)
    
    def print_result(self, result):
        """打印打分结果"""
        print("\n" + "="*80)
        print(f"标题: {result['title']}")
        print("="*80)
        print(f"创造力评分: {result['creativity_score']:.1f}/10")
        print(f"情感吸引力: {result['emotional_appeal']:.1f}/10")
        print(f"新颖度评分: {result['novelty_score']:.1f}/10")
        
        if 'predicted_engagement_rate' in result:
            eng = result['predicted_engagement_rate']
            print(f"预期互动率: {eng:.4f} ({eng*100:.2f}%)")
        
        # 特征详解
        f = result['features']
        print(f"\n特征详解:")
        print(f"  • 长度: {f['title_length']} 字符, {f['word_count']} 个词")
        print(f"  • 大小写: {f['uppercase_ratio']*100:.1f}%大写")
        print(f"  • 标点: {f['exclamation']}个!, {f['question']}个?, {f['emoji_count']}个Emoji")
        print(f"  • 密度: {f['punctuation_density']:.2f} 标点/词")


if __name__ == "__main__":
    import sys
    
    scorer = TitleScorer()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'interactive':
            scorer.interactive_mode()
        elif sys.argv[1] == 'batch':
            if len(sys.argv) > 2:
                with open(sys.argv[2], 'r', encoding='utf-8') as f:
                    titles = [line.strip() for line in f if line.strip()]
                results = scorer.compare_titles(titles)
        else:
            # 打分单个标题
            title = ' '.join(sys.argv[1:])
            result = scorer.score(title)
            scorer.print_result(result)
            suggestions = scorer.generate_suggestions(title, result)
            print("\n改进建议:")
            for s in suggestions:
                print(f"  {s}")
    else:
        # 交互模式
        scorer.interactive_mode()
