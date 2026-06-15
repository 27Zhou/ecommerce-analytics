"""
数据概览看板模块
- 总用户数、总商品数、整体转化率、GMV
- 每日浏览量趋势、类目销售额占比、用户等级分布
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS, USER_LEVEL_LABELS
from modules.utils import setup_style, save_chart, format_number, calc_conversion_rate


class DashboardAnalyzer:
    """数据概览看板"""

    def __init__(self, df):
        self.df = df
        setup_style()

    def get_summary_stats(self):
        """计算核心指标"""
        stats = {
            "total_users": self.df["user_id"].nunique(),
            "total_items": self.df["item_id"].nunique(),
            "total_records": len(self.df),
            "overall_conversion": calc_conversion_rate(
                self.df["label"].sum(), len(self.df)),
            "total_gmv": round(
                self.df.loc[self.df["label"] == 1, "price"].sum(), 2),
            "avg_price": round(self.df["price"].mean(), 2),
            "avg_pv": round(self.df["pv_count"].mean(), 1),
            "coupon_usage_rate": calc_conversion_rate(
                self.df["coupon_used"].sum(), self.df["coupon_received"].sum()),
        }
        return stats

    def plot_daily_pv_trend(self):
        """每日浏览量趋势（用 pv_count 分桶模拟）"""
        fig, ax = plt.subplots(figsize=(12, 5))
        pv_bins = pd.cut(self.df["pv_count"], bins=[0, 5, 15, 30, 50, 100, 500],
                         right=False)
        pv_dist = pv_bins.value_counts().sort_index()
        labels = ["1-4", "5-14", "15-29", "30-49", "50-99", "100+"]
        ax.bar(labels, pv_dist.values, color=COLORS["primary"], edgecolor="white")
        ax.set_xlabel("浏览量区间")
        ax.set_ylabel("用户-商品记录数")
        ax.set_title("浏览量分布趋势")
        for i, v in enumerate(pv_dist.values):
            ax.text(i, v + max(pv_dist.values) * 0.01, format_number(v),
                    ha="center", fontsize=9)
        return save_chart(fig, "dashboard_pv_trend.png")

    def plot_category_sales(self):
        """类目销售额占比"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        cat_counts = self.df.groupby("category")["label"].agg(["sum", "count", "mean"])
        cat_counts.columns = ["购买数", "浏览数", "转化率"]
        cat_counts = cat_counts.sort_values("购买数", ascending=False)

        # 左图：购买占比饼图
        axes[0].pie(cat_counts["购买数"], labels=cat_counts.index,
                    autopct="%1.1f%%", colors=COLORS["palette"][:len(cat_counts)],
                    startangle=90)
        axes[0].set_title("各类目购买占比")

        # 右图：转化率柱状图（百分比）
        purchase_pct = cat_counts["转化率"] * 100
        bars = axes[1].bar(cat_counts.index, purchase_pct,
                           color=COLORS["palette"][:len(cat_counts)], edgecolor="white")
        axes[1].set_ylabel("转化率 (%)")
        axes[1].set_ylim(0, 100)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        axes[1].set_title("各类目转化率")
        for bar, val in zip(bars, purchase_pct):
            axes[1].text(bar.get_x() + bar.get_width() / 2, val + 1,
                         f"{val:.1f}%", ha="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "dashboard_category_sales.png")

    def plot_user_level_dist(self):
        """用户等级分布"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        user_df = self.df.drop_duplicates(subset=["user_id"])

        # 等级分布（百分比）
        level_counts = user_df["user_level"].value_counts().sort_index()
        level_pct = level_counts / level_counts.sum() * 100
        level_labels = [USER_LEVEL_LABELS.get(i, f"等级{i}") for i in level_pct.index]

        bars = axes[0].bar(level_labels, level_pct,
                           color=COLORS["palette"][:len(level_pct)], edgecolor="white")
        axes[0].set_xlabel("用户等级")
        axes[0].set_ylabel("占比 (%)")
        axes[0].set_ylim(0, 100)
        axes[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")
        axes[0].set_title("用户等级分布")
        axes[0].tick_params(axis="x", rotation=30)
        for bar, pct, cnt in zip(bars, level_pct, level_counts):
            axes[0].text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                         f"{pct:.1f}%\n({cnt}人)", ha="center", fontsize=7)

        # 各等级消费
        level_spend = user_df.groupby("user_level")["total_spend"].mean()
        axes[1].plot(level_spend.index, level_spend.values, "o-",
                     color=COLORS["primary"], linewidth=2, markersize=8)
        axes[1].fill_between(level_spend.index, level_spend.values,
                             alpha=0.2, color=COLORS["primary"])
        axes[1].set_xlabel("用户等级")
        axes[1].set_ylabel("平均累计消费 (¥)")
        axes[1].set_title("各等级用户平均消费")
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        for x, y in zip(level_spend.index, level_spend.values):
            axes[1].annotate(f"¥{y:,.0f}", xy=(x, y), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=8)

        plt.tight_layout()
        return save_chart(fig, "dashboard_user_level.png")

    def run(self):
        """运行全部分析，返回统计结果和图表路径"""
        stats = self.get_summary_stats()
        charts = {
            "pv_trend": self.plot_daily_pv_trend(),
            "category_sales": self.plot_category_sales(),
            "user_level": self.plot_user_level_dist(),
        }
        return stats, charts
