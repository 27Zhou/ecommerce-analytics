"""
优惠券策略分析模块
- 券使用率分析
- 按折扣×用户分层做 A/B 测试矩阵
- 领券未用用户再触达策略
- 输出分析结论和运营建议
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS
from modules.utils import setup_style, save_chart, calc_conversion_rate, get_rfm_level


class CouponAnalyzer:
    """优惠券策略分析"""

    def __init__(self, df):
        self.df = df.copy()
        self.df["rfm_level"] = self.df.apply(get_rfm_level, axis=1)
        setup_style()

    def plot_usage_rate(self):
        """券使用率分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 整体使用率
        received = self.df["coupon_received"].sum()
        used = self.df["coupon_used"].sum()
        not_used = received - used
        never_received = len(self.df) - received

        labels = ["未领券", "领券未用", "领券已用"]
        values = [never_received, not_used, used]
        colors = [COLORS["light"], COLORS["warning"], COLORS["success"]]
        axes[0].pie(values, labels=labels, autopct="%1.1f%%", colors=colors, startangle=90)
        axes[0].set_title("优惠券整体使用情况")

        # 各类目券使用率
        cat_coupon = self.df[self.df["coupon_received"] == 1].groupby("category").agg(
            received=("coupon_received", "sum"),
            used=("coupon_used", "sum"),
        )
        cat_coupon["usage_rate"] = cat_coupon["used"] / cat_coupon["received"] * 100
        cat_coupon = cat_coupon.sort_values("usage_rate", ascending=True)

        axes[1].barh(cat_coupon.index, cat_coupon["usage_rate"], color=COLORS["primary"])
        axes[1].set_xlabel("券使用率 (%)")
        axes[1].set_xlim(0, 100)
        axes[1].grid(axis="x", alpha=0.3, linestyle="--")
        axes[1].set_title("各类目优惠券使用率")
        for i, v in enumerate(cat_coupon["usage_rate"]):
            axes[1].text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "coupon_usage_rate.png")

    def plot_ab_test_matrix(self):
        """A/B 测试矩阵：折扣档位 × 用户分层"""
        fig, ax = plt.subplots(figsize=(10, 6))

        disc_bins = [0, 0.1, 0.15, 0.2, 0.3, 1.0]
        disc_labels = ["0-10%", "10-15%", "15-20%", "20-30%", "30%+"]
        self.df["disc_level"] = pd.cut(self.df["discount_rate"], bins=disc_bins, labels=disc_labels, right=False)

        # 构建矩阵
        matrix = self.df.groupby(["rfm_level", "disc_level"], observed=False)["label"].mean() * 100
        matrix = matrix.unstack(fill_value=0)

        sns.heatmap(matrix, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                    linewidths=0.5, cbar_kws={"label": "购买率 (%)"})
        ax.set_xlabel("折扣档位")
        ax.set_ylabel("用户分层")
        ax.set_title("A/B 测试矩阵：折扣 × 用户分层 购买率 (%)")
        plt.tight_layout()
        return save_chart(fig, "coupon_ab_matrix.png")

    def plot_unused_coupon_users(self):
        """领券未用用户画像"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        unused = self.df[(self.df["coupon_received"] == 1) & (self.df["coupon_used"] == 0)]
        used = self.df[self.df["coupon_used"] == 1]

        # 年龄对比
        axes[0].hist(unused["age"], bins=20, alpha=0.6, color=COLORS["warning"], label="未用券", density=True)
        axes[0].hist(used["age"], bins=20, alpha=0.6, color=COLORS["success"], label="已用券", density=True)
        axes[0].set_xlabel("年龄")
        axes[0].set_ylabel("密度")
        axes[0].set_title("领券未用 vs 已用：年龄分布")
        axes[0].legend()

        # 等级对比
        unused_level = unused["user_level"].value_counts().sort_index()
        used_level = used["user_level"].value_counts().sort_index()
        x = np.arange(len(unused_level))
        w = 0.35
        axes[1].bar(x - w / 2, unused_level.values, w, color=COLORS["warning"], label="未用券")
        axes[1].bar(x + w / 2, used_level.reindex(unused_level.index, fill_value=0).values, w,
                    color=COLORS["success"], label="已用券")
        axes[1].set_xlabel("用户等级")
        axes[1].set_ylabel("人数")
        axes[1].set_title("领券未用 vs 已用：等级分布")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(unused_level.index)
        axes[1].legend()

        # 消费频次对比
        axes[2].hist(unused["purchase_freq"], bins=20, alpha=0.6, color=COLORS["warning"], label="未用券", density=True)
        axes[2].hist(used["purchase_freq"], bins=20, alpha=0.6, color=COLORS["success"], label="已用券", density=True)
        axes[2].set_xlabel("购买频次")
        axes[2].set_ylabel("密度")
        axes[2].set_title("领券未用 vs 已用：购买频次分布")
        axes[2].legend()

        plt.tight_layout()
        return save_chart(fig, "coupon_unused_users.png")

    def get_coupon_stats(self):
        """返回优惠券统计数据"""
        received = int(self.df["coupon_received"].sum())
        used = int(self.df["coupon_used"].sum())
        unused_users = self.df[(self.df["coupon_received"] == 1) & (self.df["coupon_used"] == 0)]

        # 最优折扣区间
        disc_bins = [0, 0.1, 0.15, 0.2, 0.3, 1.0]
        disc_labels = ["0-10%", "10-15%", "15-20%", "20-30%", "30%+"]
        self.df["disc_level"] = pd.cut(self.df["discount_rate"], bins=disc_bins, labels=disc_labels, right=False)
        disc_purchase = self.df.groupby("disc_level", observed=False)["label"].mean()
        best_disc = disc_purchase.idxmax() if len(disc_purchase) > 0 else "N/A"

        stats = {
            "total_received": received,
            "total_used": used,
            "usage_rate": calc_conversion_rate(used, received),
            "unused_user_count": len(unused_users),
            "best_discount_range": str(best_disc),
            "best_discount_rate": round(disc_purchase.max() * 100, 2) if len(disc_purchase) > 0 else 0,
            "unused_user_profile": {
                "avg_age": round(unused_users["age"].mean(), 1) if len(unused_users) > 0 else 0,
                "avg_level": round(unused_users["user_level"].mean(), 1) if len(unused_users) > 0 else 0,
                "avg_freq": round(unused_users["purchase_freq"].mean(), 1) if len(unused_users) > 0 else 0,
                "top_level": unused_users["rfm_level"].value_counts().to_dict() if len(unused_users) > 0 else {},
            },
        }
        return stats

    def get_retention_suggestions(self):
        """生成再触达策略建议"""
        unused = self.df[(self.df["coupon_received"] == 1) & (self.df["coupon_used"] == 0)]
        if len(unused) == 0:
            return "无领券未用用户数据"

        suggestions = []
        for level in unused["rfm_level"].unique():
            subset = unused[unused["rfm_level"] == level]
            count = len(subset)
            if level == "高价值用户":
                suggestions.append(f"高价值用户 ({count}人): 专属客服 1v1 提醒，延长券有效期")
            elif level == "一般用户":
                suggestions.append(f"一般用户 ({count}人): 到期前24h推送提醒消息")
            else:
                suggestions.append(f"流失风险用户 ({count}人): 追加折扣，换更高面额券再推一次")
        return suggestions

    def run(self):
        """运行全部分析"""
        stats = self.get_coupon_stats()
        suggestions = self.get_retention_suggestions()
        stats["retention_suggestions"] = suggestions
        charts = {
            "usage_rate": self.plot_usage_rate(),
            "ab_matrix": self.plot_ab_test_matrix(),
            "unused_users": self.plot_unused_coupon_users(),
        }
        return stats, charts
