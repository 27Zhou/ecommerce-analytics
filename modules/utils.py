"""
公共工具模块 - 通用函数
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np
import os
from datetime import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS


def setup_chinese_font():
    """配置 Matplotlib 中文显示"""
    # 优先使用微软雅黑（支持更多符号）
    font_paths = [
        ("C:/Windows/Fonts/msyh.ttc", "Microsoft YaHei"),
        ("C:/Windows/Fonts/simhei.ttf", "SimHei"),
    ]

    for font_path, font_name in font_paths:
        if os.path.exists(font_path):
            fm.fontManager.addfont(font_path)
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [font_name]
            break

    plt.rcParams["axes.unicode_minus"] = False


def setup_style():
    """配置图表全局样式"""
    sns.set_theme(style="whitegrid", palette=COLORS["palette"])
    # 必须在 set_theme 之后设置中文字体，否则会被覆盖
    setup_chinese_font()
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 100


def save_chart(fig, filename, output_dir=None):
    """保存图表到文件"""
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    fig.savefig(filepath, bbox_inches="tight", dpi=150, facecolor="white")
    plt.close(fig)
    return filepath


def format_number(n):
    """格式化数字显示"""
    if n >= 10000:
        return f"{n/10000:.1f}万"
    elif n >= 1000:
        return f"{n/1000:.1f}k"
    return str(int(n))


def calc_conversion_rate(numerator, denominator):
    """计算转化率，避免除零"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator * 100, 2)


def get_rfm_level(row):
    """RFM 分层逻辑
    R (Recency): 最近活跃间隔，越小越好
    F (Frequency): 购买频次，越大越好
    M (Monetary): 消费金额，越大越好
    评分: 1=最优, 3=最差
    """
    # R: 间隔越短越好
    r = row.get("last_click_gap", 999)
    r_score = 1 if r <= 3 else (2 if r <= 14 else 3)

    # F: 频次越高越好
    f = row.get("purchase_freq", 0)
    f_score = 1 if f >= 15 else (2 if f >= 5 else 3)

    # M: 金额越高越好
    m = row.get("total_spend", 0)
    m_score = 1 if m >= 5000 else (2 if m >= 1500 else 3)

    total = r_score + f_score + m_score
    if total <= 4:
        return "高价值用户"
    elif total <= 7:
        return "一般用户"
    else:
        return "流失风险用户"


def timestamp():
    """返回当前时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
