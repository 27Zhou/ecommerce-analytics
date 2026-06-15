"""
用户画像分析模块
- 年龄分布、性别比例分析
- RFM 分层（高价值/一般/流失风险）
- 用户价值分层（高价值/潜力/低活跃）
- 粉丝数、关注数分布
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS
from modules.utils import setup_style, save_chart, get_rfm_level


def _add_bar_labels(ax, bars, fmt="{:.0f}", offset=0.02):
    """给柱状图添加数据标签"""
    max_h = max(b.get_height() for b in bars)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + max_h * offset,
                fmt.format(h), ha="center", fontsize=9)


class UserAnalyzer:
    """用户画像分析"""

    def __init__(self, df):
        self.df = df
        self.user_df = df.drop_duplicates(subset=["user_id"]).copy()
        setup_style()

    def plot_age_distribution(self):
        """年龄分布柱状图"""
        fig, ax = plt.subplots(figsize=(10, 5))
        bins = [0, 18, 25, 35, 45, 55, 100]
        labels = ["<18", "18-24", "25-34", "35-44", "45-54", "55+"]
        self.user_df["age_group"] = pd.cut(self.user_df["age"], bins=bins, labels=labels, right=False)
        age_counts = self.user_df["age_group"].value_counts().reindex(labels)
        age_pct = age_counts / age_counts.sum() * 100

        bars = ax.bar(labels, age_pct, color=COLORS["palette"][:len(labels)], edgecolor="white")
        ax.set_xlabel("年龄段")
        ax.set_ylabel("占比 (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title("用户年龄分布")

        for bar, pct, cnt in zip(bars, age_pct, age_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 1,
                    f"{pct:.1f}%\n({cnt}人)", ha="center", fontsize=8)

        return save_chart(fig, "user_age_dist.png")

    def plot_gender_ratio(self):
        """性别比例饼图"""
        fig, ax = plt.subplots(figsize=(6, 6))
        gender_map = {0: "女", 1: "男"}
        gender_counts = self.user_df["gender"].map(gender_map).value_counts()
        colors = [COLORS["danger"], COLORS["primary"]]
        ax.pie(gender_counts.values, labels=gender_counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, explode=[0.02] * len(gender_counts))
        ax.set_title("用户性别比例")
        return save_chart(fig, "user_gender_ratio.png")

    def plot_rfm_segmentation(self):
        """RFM 分层"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        self.user_df["rfm_level"] = self.user_df.apply(get_rfm_level, axis=1)
        rfm_counts = self.user_df["rfm_level"].value_counts()
        rfm_pct = rfm_counts / rfm_counts.sum() * 100

        color_map = {"高价值用户": COLORS["success"], "一般用户": COLORS["warning"],
                     "流失风险用户": COLORS["danger"]}
        bar_colors = [color_map.get(x, COLORS["light"]) for x in rfm_pct.index]

        # 左图：占比
        bars1 = axes[0].bar(rfm_pct.index, rfm_pct, color=bar_colors, edgecolor="white")
        axes[0].set_ylabel("占比 (%)")
        axes[0].set_ylim(0, 100)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")
        axes[0].set_title("RFM 用户分层")
        for bar, pct, cnt in zip(bars1, rfm_pct, rfm_counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2, pct + 1,
                         f"{pct:.1f}%\n({cnt}人)", ha="center", fontsize=8)

        # 右图：各层级消费
        rfm_spend = self.user_df.groupby("rfm_level")["total_spend"].mean()
        bar_colors2 = [color_map.get(x, COLORS["light"]) for x in rfm_spend.index]
        bars2 = axes[1].bar(rfm_spend.index, rfm_spend.values, color=bar_colors2, edgecolor="white")
        axes[1].set_ylabel("平均累计消费 (¥)")
        axes[1].set_title("各层级平均消费")
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        for bar, val in zip(bars2, rfm_spend.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(rfm_spend.values) * 0.02,
                         f"¥{val:,.0f}", ha="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "user_rfm.png")

    def plot_social_distribution(self):
        """粉丝数、关注数分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 粉丝数分布
        fan_bins = [0, 1, 5, 10, 20, 50, 9999]
        fan_labels = ["0", "1-4", "5-9", "10-19", "20-49", "50+"]
        self.user_df["fan_group"] = pd.cut(self.user_df["fans_num"], bins=fan_bins, labels=fan_labels, right=False)
        fan_counts = self.user_df["fan_group"].value_counts().reindex(fan_labels)
        fan_pct = fan_counts / fan_counts.sum() * 100

        bars1 = axes[0].bar(fan_labels, fan_pct, color=COLORS["info"], edgecolor="white")
        axes[0].set_xlabel("粉丝数区间")
        axes[0].set_ylabel("占比 (%)")
        axes[0].set_ylim(0, 100)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")
        axes[0].set_title("粉丝数分布")
        for bar, pct in zip(bars1, fan_pct):
            axes[0].text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                         f"{pct:.1f}%", ha="center", fontsize=8)

        # 关注数分布
        follow_bins = [0, 5, 15, 30, 50, 100, 9999]
        follow_labels = ["0-4", "5-14", "15-29", "30-49", "50-99", "100+"]
        self.user_df["follow_group"] = pd.cut(self.user_df["follow_num"], bins=follow_bins, labels=follow_labels, right=False)
        follow_counts = self.user_df["follow_group"].value_counts().reindex(follow_labels)
        follow_pct = follow_counts / follow_counts.sum() * 100

        bars2 = axes[1].bar(follow_labels, follow_pct, color=COLORS["primary"], edgecolor="white")
        axes[1].set_xlabel("关注数区间")
        axes[1].set_ylabel("占比 (%)")
        axes[1].set_ylim(0, 100)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        axes[1].set_title("关注数分布")
        for bar, pct in zip(bars2, follow_pct):
            axes[1].text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                         f"{pct:.1f}%", ha="center", fontsize=8)

        plt.tight_layout()
        return save_chart(fig, "user_social_dist.png")

    def plot_user_value_segmentation(self):
        """用户价值分层：高价值/潜力/低活跃"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 用户价值评分
        # 购买频次0.45 + 消费金额0.3 + 活跃度0.25
        # 购买频次权重最高（复购是核心指标）
        freq_norm = self.user_df["purchase_freq"] / max(self.user_df["purchase_freq"].max(), 1)
        spend_norm = self.user_df["total_spend"] / max(self.user_df["total_spend"].max(), 1)
        active_norm = 1 - (self.user_df["register_days"] / max(self.user_df["register_days"].max(), 1))

        self.user_df["value_score"] = (freq_norm * 0.45 + spend_norm * 0.30 + active_norm * 0.25) * 100

        # 分层
        self.user_df["value_level"] = pd.cut(
            self.user_df["value_score"],
            bins=[0, 30, 60, 100],
            labels=["低活跃用户", "潜力用户", "高价值用户"]
        )

        level_counts = self.user_df["value_level"].value_counts()
        level_pct = level_counts / level_counts.sum() * 100

        color_map = {"高价值用户": "#27AE60", "潜力用户": "#F39C12", "低活跃用户": "#E74C3C"}
        colors = [color_map.get(x, "#BDC3C7") for x in level_pct.index]

        # 左图：占比饼图
        axes[0].pie(level_pct.values, labels=level_pct.index, autopct="%1.1f%%",
                    colors=colors, startangle=90)
        axes[0].set_title("用户价值分层")

        # 右图：各层级消费柱状图
        level_spend = self.user_df.groupby("value_level", observed=False)["total_spend"].mean()
        colors2 = [color_map.get(x, "#BDC3C7") for x in level_spend.index]
        bars = axes[1].bar(level_spend.index, level_spend.values, color=colors2, edgecolor="white")
        axes[1].set_ylabel("平均累计消费 (¥)")
        axes[1].set_title("各层级平均消费")
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, level_spend.values):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + max(level_spend.values) * 0.02,
                         f"¥{val:,.0f}", ha="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "user_value_segmentation.png")

    def get_user_stats(self):
        """返回用户画像统计数据"""
        self.user_df["rfm_level"] = self.user_df.apply(get_rfm_level, axis=1)
        stats = {
            "total_users": len(self.user_df),
            "age_mean": round(self.user_df["age"].mean(), 1),
            "age_median": round(self.user_df["age"].median(), 1),
            "gender_ratio": {
                "男": int((self.user_df["gender"] == 1).sum()),
                "女": int((self.user_df["gender"] == 0).sum()),
            },
            "rfm_distribution": self.user_df["rfm_level"].value_counts().to_dict(),
            "avg_purchase_freq": round(self.user_df["purchase_freq"].mean(), 1),
            "avg_total_spend": round(self.user_df["total_spend"].mean(), 2),
            "avg_fans": round(self.user_df["fans_num"].mean(), 1),
            "avg_follow": round(self.user_df["follow_num"].mean(), 1),
        }
        return stats

    def run(self):
        """运行全部分析"""
        stats = self.get_user_stats()

        # 用户价值分层
        self.plot_user_value_segmentation()
        value_counts = self.user_df["value_level"].value_counts().to_dict()
        stats["value_segmentation"] = value_counts

        charts = {
            "age_dist": self.plot_age_distribution(),
            "gender_ratio": self.plot_gender_ratio(),
            "rfm": self.plot_rfm_segmentation(),
            "social": self.plot_social_distribution(),
            "value_segmentation": self.plot_user_value_segmentation(),
        }
        return stats, charts
