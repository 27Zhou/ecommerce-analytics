"""
商品表现分析模块
- 类目热度、价格区间转化率分析
- 折扣率 vs 购买率曲线
- 视频/图片影响分析
- 标题情感分分析
- 商品热度指数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS
from modules.utils import setup_style, save_chart, calc_conversion_rate


def _add_bar_labels(ax, bars, fmt="{:.1f}%", offset=0.5):
    """给柱状图添加数据标签"""
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                fmt.format(h), ha="center", fontsize=9, fontweight="bold")


def _set_pct_axis(ax, max_pct=100):
    """设置百分比坐标轴"""
    ax.set_ylim(0, max_pct)
    ax.set_ylabel("百分比 (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(axis="y", alpha=0.3, linestyle="--")


class ProductAnalyzer:
    """商品表现分析"""

    def __init__(self, df):
        self.df = df
        setup_style()

    def plot_category_heatmap(self):
        """类目热度：浏览量、加购率、购买率"""
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))

        cat_stats = self.df.groupby("category").agg(
            pv_total=("pv_count", "sum"),
            add2cart_rate=("add2cart", "mean"),
            purchase_rate=("label", "mean"),
        ).sort_values("pv_total", ascending=False)

        # 浏览量（数值）
        bars1 = axes[0].barh(cat_stats.index, cat_stats["pv_total"], color=COLORS["primary"])
        axes[0].set_xlabel("总浏览量")
        axes[0].set_title("各类目浏览量")
        axes[0].invert_yaxis()
        for bar, val in zip(bars1, cat_stats["pv_total"]):
            axes[0].text(val + max(cat_stats["pv_total"]) * 0.02, bar.get_y() + bar.get_height() / 2,
                         f"{val:,.0f}", va="center", fontsize=9)

        # 加购率（百分比）
        add2cart_pct = cat_stats["add2cart_rate"] * 100
        bars2 = axes[1].barh(cat_stats.index, add2cart_pct, color=COLORS["warning"])
        axes[1].set_xlabel("加购率 (%)")
        axes[1].set_title("各类目加购率")
        axes[1].set_xlim(0, 100)
        axes[1].invert_yaxis()
        for bar, val in zip(bars2, add2cart_pct):
            axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=9)

        # 购买率（百分比）
        purchase_pct = cat_stats["purchase_rate"] * 100
        bars3 = axes[2].barh(cat_stats.index, purchase_pct, color=COLORS["success"])
        axes[2].set_xlabel("购买率 (%)")
        axes[2].set_title("各类目购买率")
        axes[2].set_xlim(0, 100)
        axes[2].invert_yaxis()
        for bar, val in zip(bars3, purchase_pct):
            axes[2].text(val + 1, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "product_category_heatmap.png")

    def plot_price_conversion(self):
        """价格区间转化率"""
        fig, ax = plt.subplots(figsize=(12, 5))
        price_bins = [0, 30, 60, 100, 150, 200, 300, 500, 2000]
        price_labels = ["<30", "30-59", "60-99", "100-149", "150-199", "200-299", "300-499", "500+"]
        self.df["price_range"] = pd.cut(self.df["price"], bins=price_bins, labels=price_labels, right=False)

        price_stats = self.df.groupby("price_range", observed=False).agg(
            count=("label", "count"),
            purchase_rate=("label", "mean"),
        )
        price_stats["purchase_rate"] = price_stats["purchase_rate"] * 100

        # 双轴图
        ax2 = ax.twinx()
        bars = ax.bar(price_labels, price_stats["count"], color=COLORS["light"], alpha=0.6, label="记录数")
        line = ax2.plot(price_labels, price_stats["purchase_rate"], "o-",
                        color=COLORS["danger"], linewidth=2, markersize=8, label="购买率")

        ax.set_xlabel("价格区间 (¥)")
        ax.set_ylabel("记录数")
        ax2.set_ylabel("购买率 (%)")
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title("价格区间 vs 购买率")

        # 标注最高转化率
        max_idx = price_stats["purchase_rate"].idxmax()
        max_val = price_stats["purchase_rate"].max()
        ax2.annotate(f"最高: {max_val:.1f}%",
                     xy=(list(price_labels).index(max_idx), max_val),
                     xytext=(list(price_labels).index(max_idx) + 1, max_val + 5),
                     arrowprops=dict(arrowstyle="->", color="red"), color="red", fontsize=10, ha="center")

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        return save_chart(fig, "product_price_conversion.png")

    def plot_discount_curve(self):
        """折扣率 vs 购买率曲线"""
        fig, ax = plt.subplots(figsize=(12, 5))
        disc_bins = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 1.0]
        disc_labels = ["0-4%", "5-9%", "10-14%", "15-19%", "20-24%", "25-29%",
                       "30-34%", "35-39%", "40-49%", "50%+"]
        self.df["disc_range"] = pd.cut(self.df["discount_rate"], bins=disc_bins, labels=disc_labels, right=False)

        disc_stats = self.df.groupby("disc_range", observed=False).agg(
            count=("label", "count"),
            purchase_rate=("label", "mean"),
        )
        disc_stats["purchase_rate"] = disc_stats["purchase_rate"] * 100

        ax2 = ax.twinx()
        ax.bar(disc_labels, disc_stats["count"], color=COLORS["light"], alpha=0.6, label="记录数")
        ax2.plot(disc_labels, disc_stats["purchase_rate"], "s-",
                 color=COLORS["success"], linewidth=2, markersize=8, label="购买率")

        ax.set_xlabel("折扣率区间")
        ax.set_ylabel("记录数")
        ax2.set_ylabel("购买率 (%)")
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title("折扣率 vs 购买率")
        ax.tick_params(axis="x", rotation=30)

        # 标注最优区间
        max_idx = disc_stats["purchase_rate"].idxmax()
        max_val = disc_stats["purchase_rate"].max()
        ax2.annotate(f"最优: {max_idx}\n({max_val:.1f}%)",
                     xy=(list(disc_labels).index(max_idx), max_val),
                     xytext=(list(disc_labels).index(max_idx) + 1, max_val + 5),
                     arrowprops=dict(arrowstyle="->", color="green"), color="green", fontsize=10)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        return save_chart(fig, "product_discount_curve.png")

    def plot_content_impact(self):
        """视频/图片对购买率的影响"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 视频影响
        video_stats = self.df.groupby("has_video").agg(
            purchase_rate=("label", "mean"),
            count=("label", "count"),
        )
        video_stats.index = ["无视频", "有视频"]
        video_pct = video_stats["purchase_rate"] * 100

        bars = axes[0].bar(video_stats.index, video_pct,
                           color=[COLORS["light"], COLORS["primary"]], edgecolor="white")
        _set_pct_axis(axes[0])
        axes[0].set_title("视频对购买率的影响")
        _add_bar_labels(axes[0], bars)

        # 图片数量影响
        img_stats = self.df.groupby("img_count").agg(
            purchase_rate=("label", "mean"),
            count=("label", "count"),
        )
        img_pct = img_stats["purchase_rate"] * 100

        axes[1].plot(img_stats.index, img_pct, "o-",
                     color=COLORS["info"], linewidth=2, markersize=8)
        axes[1].fill_between(img_stats.index, img_pct, alpha=0.2, color=COLORS["info"])
        axes[1].set_xlabel("图片数量")
        axes[1].set_ylabel("购买率 (%)")
        axes[1].set_ylim(0, 100)
        axes[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        axes[1].grid(axis="y", alpha=0.3, linestyle="--")
        axes[1].set_title("图片数量 vs 购买率")

        # 添加数据标签
        for x, y in zip(img_stats.index, img_pct):
            axes[1].annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 8),
                             textcoords="offset points", ha="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "product_content_impact.png")

    def plot_title_sentiment(self):
        """标题情感分 vs 购买率"""
        fig, ax = plt.subplots(figsize=(12, 5))
        sent_bins = [0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        sent_labels = ["<0.3", "0.3-0.5", "0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9+"]
        self.df["sent_range"] = pd.cut(self.df["title_emo_score"], bins=sent_bins, labels=sent_labels, right=False)

        sent_stats = self.df.groupby("sent_range", observed=False).agg(
            purchase_rate=("label", "mean"),
            count=("label", "count"),
        )
        sent_pct = sent_stats["purchase_rate"] * 100

        ax2 = ax.twinx()
        ax.bar(sent_labels, sent_stats["count"], color=COLORS["light"], alpha=0.6, label="记录数")
        ax2.plot(sent_labels, sent_pct, "D-",
                 color=COLORS["danger"], linewidth=2, markersize=8, label="购买率")

        ax.set_xlabel("标题情感分区间")
        ax.set_ylabel("记录数")
        ax2.set_ylabel("购买率 (%)")
        ax2.set_ylim(0, 100)
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax2.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title("标题情感分 vs 购买率")
        ax.tick_params(axis="x", rotation=30)

        # 添加数据标签
        for x, y in zip(sent_labels, sent_pct):
            ax2.annotate(f"{y:.1f}%", xy=(x, y), xytext=(0, 8),
                         textcoords="offset points", ha="center", fontsize=9)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        return save_chart(fig, "product_title_sentiment.png")

    def plot_hotness_index(self):
        """商品热度指数分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 计算热度指数
        # 浏览0.1 + 点赞0.15 + 收藏0.2 + 评论0.2 + 购买0.35
        # 购买权重最高（含金量最高），浏览权重最低（门槛最低）
        item_stats = self.df.groupby("item_id").agg(
            pv=("pv_count", "sum"),
            collect=("collect_num", "sum"),
            like=("like_num", "sum"),
            comment=("comment_num", "sum"),
            purchase=("label", "sum"),
        ).reset_index()

        for col in ["pv", "collect", "like", "comment", "purchase"]:
            max_val = item_stats[col].max()
            if max_val > 0:
                item_stats[f"{col}_norm"] = item_stats[col] / max_val
            else:
                item_stats[f"{col}_norm"] = 0

        item_stats["hotness"] = (
            item_stats["pv_norm"] * 0.10 +
            item_stats["like_norm"] * 0.15 +
            item_stats["collect_norm"] * 0.20 +
            item_stats["comment_norm"] * 0.20 +
            item_stats["purchase_norm"] * 0.35
        ) * 100

        # Top10 热门商品
        top10 = item_stats.nlargest(10, "hotness")
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
        bars = axes[0].barh(range(len(top10)), top10["hotness"], color=colors)
        axes[0].set_yticks(range(len(top10)))
        axes[0].set_yticklabels([f"商品{i[-4:]}" for i in top10["item_id"]], fontsize=9)
        axes[0].set_xlabel("热度指数 (0-100)")
        axes[0].set_title("Top10 热门商品")
        axes[0].set_xlim(0, 100)
        axes[0].invert_yaxis()
        axes[0].grid(axis="x", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, top10["hotness"]):
            axes[0].text(val + 1, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}", va="center", fontsize=9)

        # 热度分布
        axes[1].hist(item_stats["hotness"], bins=30, color=COLORS["primary"], edgecolor="white")
        axes[1].axvline(item_stats["hotness"].mean(), color=COLORS["danger"], linestyle="--",
                        label=f"均值: {item_stats['hotness'].mean():.1f}")
        axes[1].set_xlabel("热度指数 (0-100)")
        axes[1].set_ylabel("商品数")
        axes[1].set_title("商品热度分布")
        axes[1].set_xlim(0, 100)
        axes[1].legend()

        plt.tight_layout()
        return save_chart(fig, "product_hotness.png"), item_stats

    def get_product_stats(self):
        """返回商品分析统计数据"""
        cat_stats = self.df.groupby("category").agg(
            count=("label", "count"),
            purchase_rate=("label", "mean"),
            avg_price=("price", "mean"),
        )
        stats = {
            "category_stats": cat_stats.to_dict(),
            "avg_price": round(self.df["price"].mean(), 2),
            "avg_discount": round(self.df["discount_rate"].mean(), 3),
            "video_impact": {
                "有视频购买率": round(self.df[self.df["has_video"] == 1]["label"].mean() * 100, 2),
                "无视频购买率": round(self.df[self.df["has_video"] == 0]["label"].mean() * 100, 2),
            },
        }
        return stats

    def run(self):
        """运行全部分析"""
        stats = self.get_product_stats()

        # 热度指数分析
        hotness_chart, hotness_data = self.plot_hotness_index()
        top10_hot = hotness_data.nlargest(10, "hotness")[["item_id", "hotness"]].to_dict("records")
        stats["top10_hot_items"] = top10_hot
        stats["avg_hotness"] = round(hotness_data["hotness"].mean(), 1)

        charts = {
            "category_heatmap": self.plot_category_heatmap(),
            "price_conversion": self.plot_price_conversion(),
            "discount_curve": self.plot_discount_curve(),
            "content_impact": self.plot_content_impact(),
            "title_sentiment": self.plot_title_sentiment(),
            "hotness": hotness_chart,
        }
        return stats, charts
