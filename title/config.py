# YouTube标题质量打分器 - 配置文件

# ========== DeepSeek API 配置 ==========
DEEPSEEK_API_KEY = "sk-4ca9f88ff7894e779717b79a26120338"  # 替换为你的API密钥
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"

# 如何获取API密钥:
# 1. 访问 https://platform.deepseek.com/
# 2. 注册/登录账号
# 3. 创建API密钥
# 4. 复制粘贴到上方

# ========== 模型参数 ==========
XGBOOST_PARAMS = {
    'n_estimators': 100,      # 树的数量
    'max_depth': 6,            # 树的最大深度
    'learning_rate': 0.1,      # 学习率
    'subsample': 0.8,          # 样本采样比例
    'colsample_bytree': 0.8,   # 特征采样比例
    'random_state': 42
}

# ========== 数据配置 ==========
DATA_PATH = "youtube_video.csv"
TEST_SIZE = 0.2              # 测试集比例
DEEPSEEK_SAMPLE_SIZE = 1000  # DeepSeek评估的采样数量
API_DELAY = 0.5              # API调用间隔（秒）

# ========== 输出配置 ==========
OUTPUT_DIR = "./"
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# ========== 日志配置 ==========
VERBOSE = True
LOG_EVERY_N = 10  # 每N条记录输出一次日志
