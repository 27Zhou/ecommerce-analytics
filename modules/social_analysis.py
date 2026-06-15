"""
社交影响力分析模块
- 粉丝数分布、互动率 Top20
- 综合影响力评分（粉丝+互动+分享+转化率加权）
- 类目偏好分析
- 商品社交影响力指数
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS
from modules.utils import setup_style, save_chart


class SocialAnalyzer:
    """社交影响力分析"""

    def __init__(self, df):
        self.df = df.copy()
        self.user_df = df.drop_duplicates(subset=["user_id"]).copy()
        setup_style()

    def calc_influence_score(self):
        """计算综合影响力评分"""
        user_behavior = self.df.groupby("user_id").agg(
            avg_interaction_rate=("interaction_rate", "mean"),
            total_share=("share_num", "sum"),
            total_like=("like_num", "sum"),
            total_comment=("comment_num", "sum"),
            purchase_rate=("label", "mean"),
        ).reset_index()

        user_behavior = user_behavior.merge(
            self.user_df[["user_id", "fans_num", "follow_num"]],
            on="user_id", how="left"
        )

        # 影响力评分 = 粉丝0.15 + 互动率0.25 + 分享0.3 + 转化率0.3
        # 分享权重提升（社交传播核心），粉丝权重降低（容易刷）
        user_behavior["influence_score"] = (
            user_behavior["fans_num"].clip(0, 10000) / 10000 * 15 +
            user_behavior["avg_interaction_rate"].clip(0, 200) / 200 * 25 +
            user_behavior["total_share"].clip(0, 500) / 500 * 30 +
            user_behavior["purchase_rate"] * 30
        )

        self.influence_df = user_behavior.sort_values("influence_score", ascending=False)
        return self.influence_df

    def plot_fan_distribution(self):
        """粉丝数区间分布"""
        fig, ax = plt.subplots(figsize=(10, 5))
        fan_bins = [0, 1, 5, 10, 20, 50, 100, 9999]
        fan_labels = ["0", "1-4", "5-9", "10-19", "20-49", "50-99", "100+"]
        self.user_df["fan_group"] = pd.cut(self.user_df["fans_num"], bins=fan_bins, labels=fan_labels, right=False)
        fan_counts = self.user_df["fan_group"].value_counts().reindex(fan_labels)
        fan_pct = fan_counts / fan_counts.sum() * 100

        bars = ax.bar(fan_labels, fan_pct, color=COLORS["palette"][:len(fan_labels)], edgecolor="white")
        ax.set_xlabel("粉丝数区间")
        ax.set_ylabel("占比 (%)")
        ax.set_ylim(0, 100)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0f}%"))
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_title("用户粉丝数分布")

        for bar, pct, cnt in zip(bars, fan_pct, fan_counts):
            ax.text(bar.get_x() + bar.get_width() / 2, pct + 0.5,
                    f"{pct:.1f}%\n({cnt})", ha="center", fontsize=8)

        return save_chart(fig, "social_fan_dist.png")

    def plot_top20_interaction(self):
        """互动率 Top20"""
        fig, ax = plt.subplots(figsize=(12, 6))
        if not hasattr(self, "influence_df"):
            self.calc_influence_score()

        top20 = self.influence_df.head(20).sort_values("avg_interaction_rate")
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top20)))

        bars = ax.barh(range(len(top20)), top20["avg_interaction_rate"], color=colors)
        ax.set_yticks(range(len(top20)))
        ax.set_yticklabels([f"U{uid[-4:]}" for uid in top20["user_id"]], fontsize=8)
        ax.set_xlabel("平均互动率")
        ax.set_title("互动率 Top20 用户")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, top20["avg_interaction_rate"]):
            ax.text(val + max(top20["avg_interaction_rate"]) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", fontsize=8)

        plt.tight_layout()
        return save_chart(fig, "social_top20_interaction.png")

    def plot_influence_score(self):
        """影响力评分分布"""
        fig, ax = plt.subplots(figsize=(10, 5))
        if not hasattr(self, "influence_df"):
            self.calc_influence_score()

        ax.hist(self.influence_df["influence_score"], bins=30, color=COLORS["info"],
                edgecolor="white", alpha=0.8)
        ax.axvline(self.influence_df["influence_score"].mean(), color=COLORS["danger"],
                   linestyle="--", linewidth=2, label=f"均值: {self.influence_df['influence_score'].mean():.1f}")
        ax.set_xlabel("影响力评分 (0-100)")
        ax.set_ylabel("用户数")
        ax.set_title("影响力评分分布")
        ax.set_xlim(0, 100)
        ax.legend()
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        return save_chart(fig, "social_influence_dist.png")

    def plot_category_preference(self):
        """高影响力用户类目偏好"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        if not hasattr(self, "influence_df"):
            self.calc_influence_score()

        # Top10% 用户的类目偏好
        top10_pct = max(int(len(self.influence_df) * 0.1), 10)
        top_users = self.influence_df.head(top10_pct)["user_id"]
        top_user_data = self.df[self.df["user_id"].isin(top_users)]

        cat_pref = top_user_data["category"].value_counts()
        axes[0].pie(cat_pref.values, labels=cat_pref.index, autopct="%1.1f%%",
                    colors=COLORS["palette"][:len(cat_pref)], startangle=90)
        axes[0].set_title("Top10% 影响力用户 - 类目偏好")

        # 各类目用户的平均影响力
        user_with_cat = self.df.merge(
            self.influence_df[["user_id", "influence_score"]], on="user_id")
        cat_influence = user_with_cat.groupby("category")["influence_score"].mean().sort_values(ascending=True)

        bars = axes[1].barh(cat_influence.index, cat_influence.values, color=COLORS["primary"])
        axes[1].set_xlabel("平均影响力评分 (0-100)")
        axes[1].set_title("各类目用户平均影响力")
        axes[1].set_xlim(0, 100)
        axes[1].grid(axis="x", alpha=0.3, linestyle="--")
        for i, v in enumerate(cat_influence.values):
            axes[1].text(v + 1, i, f"{v:.1f}", va="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "social_category_preference.png")

    def plot_social_influence_index(self):
        """商品社交影响力指数分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 按商品聚合社交数据
        item_social = self.df.groupby("item_id").agg(
            like=("like_num", "sum"),
            comment=("comment_num", "sum"),
            share=("share_num", "sum"),
        ).reset_index()

        # 标准化
        for col in ["like", "comment", "share"]:
            max_val = item_social[col].max()
            if max_val > 0:
                item_social[f"{col}_norm"] = item_social[col] / max_val
            else:
                item_social[f"{col}_norm"] = 0

        # 社交影响力指数 = 点赞×0.3 + 评论×0.3 + 分享×0.4
        item_social["social_index"] = (
            item_social["like_norm"] * 0.3 +
            item_social["comment_norm"] * 0.3 +
            item_social["share_norm"] * 0.4
        ) * 100

        # 分层
        item_social["influence_level"] = pd.cut(
            item_social["social_index"],
            bins=[0, 20, 50, 100],
            labels=["低影响商品", "普通商品", "高影响商品"]
        )

        level_counts = item_social["influence_level"].value_counts()
        level_pct = level_counts / level_counts.sum() * 100

        color_map = {"高影响商品": "#27AE60", "普通商品": "#F39C12", "低影响商品": "#E74C3C"}
        colors = [color_map.get(x, "#BDC3C7") for x in level_pct.index]

        # 左图：占比饼图
        axes[0].pie(level_pct.values, labels=level_pct.index, autopct="%1.1f%%",
                    colors=colors, startangle=90)
        axes[0].set_title("商品社交影响力分层")

        # 右图：Top10 高影响商品
        top10 = item_social.nlargest(10, "social_index")
        colors2 = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))
        bars = axes[1].barh(range(len(top10)), top10["social_index"], color=colors2)
        axes[1].set_yticks(range(len(top10)))
        axes[1].set_yticklabels([f"商品{i[-4:]}" for i in top10["item_id"]], fontsize=9)
        axes[1].set_xlabel("社交影响力指数 (0-100)")
        axes[1].set_title("Top10 高影响商品")
        axes[1].set_xlim(0, 100)
        axes[1].invert_yaxis()
        axes[1].grid(axis="x", alpha=0.3, linestyle="--")
        for bar, val in zip(bars, top10["social_index"]):
            axes[1].text(val + 1, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}", va="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "social_influence_index.png"), item_social

    def get_social_stats(self):
        """返回社交分析统计"""
        if not hasattr(self, "influence_df"):
            self.calc_influence_score()

        top10 = self.influence_df.head(10)
        stats = {
            "total_influencers": len(self.influence_df[self.influence_df["influence_score"] > 50]),
            "avg_influence_score": round(self.influence_df["influence_score"].mean(), 2),
            "top10_users": top10[["user_id", "influence_score", "fans_num",
                                   "avg_interaction_rate", "purchase_rate"]].to_dict("records"),
            "high_influence_categories": self._get_top_categories(),
        }
        return stats

    def _get_top_categories(self):
        """获取高影响力用户偏好的类目"""
        top10_pct = max(int(len(self.influence_df) * 0.1), 10)
        top_users = self.influence_df.head(top10_pct)["user_id"]
        top_user_data = self.df[self.df["user_id"].isin(top_users)]
        return top_user_data["category"].value_counts().head(3).to_dict()

    def get_promotion_suggestions(self):
        """生成推广建议"""
        if not hasattr(self, "influence_df"):
            self.calc_influence_score()

        suggestions = []
        top5 = self.influence_df.head(5)
        for _, row in top5.iterrows():
            uid = row["user_id"]
            score = row["influence_score"]
            fans = row["fans_num"]

            user_cats = self.df[self.df["user_id"] == uid]["category"].value_counts()
            pref_cat = user_cats.index[0] if len(user_cats) > 0 else "未知"

            suggestions.append({
                "user_id": uid,
                "influence_score": round(score, 1),
                "fans": int(fans),
                "preferred_category": pref_cat,
                "suggestion": f"影响力评分 {score:.1f}，粉丝 {int(fans)}，偏好 {pref_cat}，建议优先合作推广该类目新品",
            })
        return suggestions

    def run(self):
        """运行全部分析"""
        self.calc_influence_score()
        stats = self.get_social_stats()
        stats["promotion_suggestions"] = self.get_promotion_suggestions()

        # 社交影响力指数
        social_chart, social_data = self.plot_social_influence_index()
        stats["social_influence_distribution"] = social_data["influence_level"].value_counts().to_dict()

        charts = {
            "fan_dist": self.plot_fan_distribution(),
            "top20_interaction": self.plot_top20_interaction(),
            "influence_dist": self.plot_influence_score(),
            "category_preference": self.plot_category_preference(),
            "social_influence_index": social_chart,
        }
        return stats, charts
