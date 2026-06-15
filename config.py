"""
社交电商数据分析系统 - 全局配置
"""
import os

# 项目路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

# 确保目录存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 数据库配置
DB_PATH = os.path.join(BASE_DIR, "ecommerce.db")

# 中文字体配置（Matplotlib）
FONT_CONFIG = {
    "font.sans-serif": ["SimHei", "Microsoft YaHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
}

# 图表配色方案
COLORS = {
    "primary": "#4A90D9",
    "success": "#27AE60",
    "warning": "#F39C12",
    "danger": "#E74C3C",
    "info": "#8E44AD",
    "light": "#BDC3C7",
    "palette": ["#4A90D9", "#27AE60", "#F39C12", "#E74C3C", "#8E44AD", "#1ABC9C", "#E67E22"],
}

# 用户等级标签
USER_LEVEL_LABELS = {
    1: "普通用户",
    2: "初级会员",
    3: "中级会员",
    4: "高级会员",
    5: "VIP会员",
    6: "SVIP会员",
    7: "至尊会员",
}

# 商品类目列表
CATEGORY_LIST = ["服饰鞋包", "数码家电", "食品生鲜", "美妆个护", "其他"]
