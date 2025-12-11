"""
使用示例和测试脚本
演示如何使用各个模块
"""

from title_scorer_tool import TitleScorer
import json

# ==================== 基础打分示例 ====================

def example_1_single_score():
    """示例1: 为单个标题打分"""
    print("\n" + "="*80)
    print("示例1: 为单个标题打分")
    print("="*80)
    
    scorer = TitleScorer()
    
    titles = [
        "Why do Human Feet Wash up on This Beach? | Fascinating Horror Shorts",
        "The ULTIMATE iPhone 15 Review - You NEED to Watch This!!!",
        "Cooking Tutorial",
    ]
    
    for title in titles:
        result = scorer.score(title)
        scorer.print_result(result)
        print()

# ==================== 对比分析示例 ====================

def example_2_comparison():
    """示例2: 对比多个标题"""
    print("\n" + "="*80)
    print("示例2: 标题对比分析")
    print("="*80)
    
    scorer = TitleScorer()
    
    # A/B测试组
    versions = [
        "iPhone 15新功能",
        "iPhone 15新功能，Apple官方推荐！",
        "🔥 iPhone 15隐藏功能大曝光！你一定不知道的5个技巧",
    ]
    
    scorer.compare_titles(versions)

# ==================== 改进建议示例 ====================

def example_3_suggestions():
    """示例3: 生成改进建议"""
    print("\n" + "="*80)
    print("示例3: 改进建议生成")
    print("="*80)
    
    scorer = TitleScorer()
    
    title = "教程"
    result = scorer.score(title)
    suggestions = scorer.generate_suggestions(title, result)
    
    print(f"\n原标题: {title}")
    print("\n改进建议:")
    for s in suggestions:
        print(f"  {s}")
    
    # 改进版
    improved_title = "你可能不知道的Python高级技巧？来看看吧！"
    result_improved = scorer.score(improved_title)
    suggestions_improved = scorer.generate_suggestions(improved_title, result_improved)
    
    print(f"\n改进后: {improved_title}")
    print(f"创造力提升: {result['creativity_score']:.1f} → {result_improved['creativity_score']:.1f}")
    print(f"情感吸引提升: {result['emotional_appeal']:.1f} → {result_improved['emotional_appeal']:.1f}")

# ==================== 批量分析示例 ====================

def example_4_batch_analysis():
    """示例4: 批量分析（模拟从CSV读取）"""
    print("\n" + "="*80)
    print("示例4: 批量分析")
    print("="*80)
    
    scorer = TitleScorer()
    
    # 模拟不同类型的标题
    titles_data = {
        "游戏类": [
            "Minecraft生存模式 第1天",
            "🎮 MINECRAFT但每死一次难度+1！我能活到第几关？",
            "我用创意模式造了一个完整城市！",
        ],
        "教育类": [
            "数学课",
            "5分钟学会微积分！完整教程",
            "你可能不知道的数学秘密 - 改变对数学的认知",
        ],
        "生活类": [
            "日常vlog",
            "我在纽约的一天 | 从工作到夜生活",
            "🗽 在纽约住一个月要多少钱？我的真实开销明细",
        ]
    }
    
    for category, titles in titles_data.items():
        print(f"\n📁 {category}:")
        print("-" * 80)
        results = scorer.compare_titles(titles)
        
        # 统计平均分
        avg_creativity = sum(r['creativity_score'] for r in results) / len(results)
        avg_emotional = sum(r['emotional_appeal'] for r in results) / len(results)
        avg_novelty = sum(r['novelty_score'] for r in results) / len(results)
        
        print(f"\n平均评分:")
        print(f"  创造力: {avg_creativity:.1f}/10")
        print(f"  情感吸引: {avg_emotional:.1f}/10")
        print(f"  新颖度: {avg_novelty:.1f}/10")

# ==================== 特征分析示例 ====================

def example_5_feature_analysis():
    """示例5: 特征详细分析"""
    print("\n" + "="*80)
    print("示例5: 特征详细分析")
    print("="*80)
    
    scorer = TitleScorer()
    
    title = "🔥 这个功能会改变你的生活！你一定想不到 | 完整教程"
    result = scorer.score(title)
    features = result['features']
    
    print(f"\n标题: {title}\n")
    print("特征分解:")
    print(f"  文本长度: {features['title_length']} 字符 (最优: 50-70)")
    print(f"  词语数: {features['word_count']} 个 (最优: 8-15)")
    print(f"  大小写比: {features['uppercase_ratio']*100:.1f}% (建议 < 20%)")
    print(f"  数字数: {features['digit_count']} 个")
    print(f"  问号: {features['question']} 个 (建议: 1-2个)")
    print(f"  感叹号: {features['exclamation']} 个 (建议: 0-2个)")
    print(f"  Emoji数: {features['emoji_count']} 个 (建议: 1-2个)")
    print(f"  标点密度: {features['punctuation_density']:.2f} (标点/词)")
    print(f"  是否含冒号: {'是' if features['has_colon'] else '否'}")
    print(f"  是否含竖线: {'是' if features['has_pipe'] else '否'}")
    print(f"\nAI评分:")
    print(f"  创造力: {features['creativity_score']:.1f}/10")
    print(f"  情感吸引力: {features['emotional_appeal']:.1f}/10")
    print(f"  新颖度: {features['novelty_score']:.1f}/10")

# ==================== 优化策略示例 ====================

def example_6_optimization_strategy():
    """示例6: 标题优化策略对比"""
    print("\n" + "="*80)
    print("示例6: 优化策略对比")
    print("="*80)
    
    scorer = TitleScorer()
    
    # 原始标题
    original = "Python教程"
    
    # 不同优化策略
    strategies = {
        "原始": original,
        "添加问号": "你真的懂Python吗？",
        "添加数字": "5个Python高级技巧",
        "添加情感": "学了10年Python才明白的秘密！",
        "组合优化": "🐍 5个Python高级技巧，你可能想不到？完整指南 | 2024必学",
    }
    
    print("\n各策略对比:")
    print("-" * 80)
    
    results_comparison = {}
    for strategy, title in strategies.items():
        result = scorer.score(title)
        results_comparison[strategy] = result
        
        print(f"\n{strategy}:")
        print(f"  标题: {title}")
        print(f"  创造力: {result['creativity_score']:.1f}")
        print(f"  情感: {result['emotional_appeal']:.1f}")
        print(f"  新颖: {result['novelty_score']:.1f}")
    
    # 汇总
    print("\n" + "="*80)
    print("🏆 最优方案:")
    best = max(results_comparison.items(), 
               key=lambda x: (x[1]['creativity_score'] + x[1]['emotional_appeal'] + x[1]['novelty_score']) / 3)
    print(f"  {best[0]}: {strategies[best[0]]}")

# ==================== 主函数 ====================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎬 YouTube标题打分器 - 使用示例")
    print("="*80)
    
    example_1_single_score()
    example_2_comparison()
    example_3_suggestions()
    example_4_batch_analysis()
    example_5_feature_analysis()
    example_6_optimization_strategy()
    
    print("\n" + "="*80)
    print("✅ 所有示例执行完毕！")
    print("="*80)
    
    print("\n💡 接下来你可以:")
    print("  1. 运行 'python title_scorer_tool.py interactive' 进入交互模式")
    print("  2. 运行 'python quick_demo.py' 查看完整分析图表")
    print("  3. 运行 'python youtube_title_scorer.py' 使用DeepSeek API（需要配置密钥）")
