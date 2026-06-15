"""
数据导入与清洗模块
- 读取 CSV 数据文件
- 处理缺失值、异常值（刷单、重复点击标记）
- 支持增量加载和 SQLite 索引优化
"""
import pandas as pd
import numpy as np
import sqlite3
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import DB_PATH


class DataLoader:
    """数据加载与清洗器"""

    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH
        self.df = None
        self.clean_log = []

    # ── 数据导入 ──────────────────────────────────────────
    def load_csv(self, filepath):
        """读取 CSV 文件"""
        self.df = pd.read_csv(filepath, encoding="utf-8")
        self.clean_log.append(f"[导入] 读取文件: {filepath}, 共 {len(self.df)} 条记录")
        return self.df

    def load_to_sqlite(self):
        """将数据加载到 SQLite 数据库"""
        if self.df is None:
            raise ValueError("请先加载数据")

        conn = sqlite3.connect(self.db_path)

        # 拆分为三张表存储
        user_cols = ["user_id", "age", "gender", "user_level", "purchase_freq",
                     "total_spend", "register_days", "follow_num", "fans_num"]
        item_cols = ["item_id", "price", "discount_rate", "category", "title_length",
                     "title_emo_score", "img_count", "has_video"]
        behavior_cols = ["user_id", "item_id", "like_num", "comment_num", "share_num",
                         "collect_num", "is_follow_author", "add2cart", "coupon_received",
                         "coupon_used", "pv_count", "last_click_gap", "interaction_rate",
                         "purchase_intent", "freshness_score", "social_influence", "label"]

        # 写入三张表（去重）
        self.df[user_cols].drop_duplicates(subset=["user_id"]).to_sql(
            "user_info", conn, if_exists="replace", index=False)
        self.df[item_cols].drop_duplicates(subset=["item_id"]).to_sql(
            "item_info", conn, if_exists="replace", index=False)
        self.df[behavior_cols].to_sql(
            "behavior_log", conn, if_exists="replace", index=False)

        # 建索引
        cursor = conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_user ON behavior_log(user_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_item ON behavior_log(item_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_label ON behavior_log(label)")
        conn.commit()
        conn.close()

        self.clean_log.append(f"[存储] 已写入 SQLite: {self.db_path}")
        self.clean_log.append(f"[索引] 已建立 user_id, item_id, label 索引")

    def incremental_load(self, new_csv_path):
        """增量加载新数据"""
        new_df = pd.read_csv(new_csv_path, encoding="utf-8")
        before_count = len(self.df) if self.df is not None else 0
        if self.df is not None:
            self.df = pd.concat([self.df, new_df], ignore_index=True)
        else:
            self.df = new_df
        after_count = len(self.df)
        self.clean_log.append(f"[增量] 加载 {new_csv_path}, 新增 {after_count - before_count} 条")
        return self.df

    # ── 数据清洗 ──────────────────────────────────────────
    def clean(self):
        """执行完整数据清洗流程"""
        if self.df is None:
            raise ValueError("请先加载数据")

        self._log_before()

        # 1. 去重
        self._remove_duplicates()

        # 2. 缺失值处理
        self._handle_missing()

        # 3. 异常值标记
        self._flag_anomalies()

        # 4. 数据类型优化
        self._optimize_types()

        self._log_after()
        return self.df

    def _log_before(self):
        self.clean_log.append(f"\n{'='*50}")
        self.clean_log.append(f"[清洗开始] 原始数据: {len(self.df)} 行, {len(self.df.columns)} 列")
        self.clean_log.append(f"[缺失值统计]")
        missing = self.df.isnull().sum()
        for col in missing[missing > 0].index:
            self.clean_log.append(f"  {col}: {missing[col]} ({missing[col]/len(self.df)*100:.1f}%)")

    def _remove_duplicates(self):
        """去除重复行"""
        before = len(self.df)
        self.df = self.df.drop_duplicates(subset=["user_id", "item_id"], keep="first")
        after = len(self.df)
        removed = before - after
        if removed > 0:
            self.clean_log.append(f"[去重] 移除 {removed} 条重复记录 (user_id + item_id)")

    def _handle_missing(self):
        """处理缺失值"""
        filled = 0
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count == 0:
                continue
            if self.df[col].dtype in ["float64", "int64"]:
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                filled += null_count
                self.clean_log.append(f"  {col}: 用中位数 {median_val} 填充 {null_count} 个缺失值")
            else:
                mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "未知"
                self.df[col] = self.df[col].fillna(mode_val)
                filled += null_count
                self.clean_log.append(f"  {col}: 用众数 '{mode_val}' 填充 {null_count} 个缺失值")
        self.clean_log.append(f"[缺失值] 共填充 {filled} 个缺失值")

    def _flag_anomalies(self):
        """标记异常值"""
        flags = []

        # 刷单嫌疑: 互动率异常高但未购买
        if "interaction_rate" in self.df.columns:
            threshold = self.df["interaction_rate"].mean() + 3 * self.df["interaction_rate"].std()
            fraud_mask = (self.df["interaction_rate"] > threshold) & (self.df["label"] == 0)
            fraud_count = fraud_mask.sum()
            self.df["is_suspected_fraud"] = fraud_mask.astype(int)
            if fraud_count > 0:
                flags.append(f"刷单嫌疑: {fraud_count} 条 (互动率>{threshold:.1f} 且未购买)")

        # 重复点击: 同一用户对同一商品的浏览次数异常高
        if "pv_count" in self.df.columns:
            pv_threshold = self.df["pv_count"].quantile(0.99)
            repeat_mask = self.df["pv_count"] > pv_threshold
            repeat_count = repeat_mask.sum()
            self.df["is_repeat_click"] = repeat_mask.astype(int)
            if repeat_count > 0:
                flags.append(f"异常高频浏览: {repeat_count} 条 (浏览量>{pv_threshold:.0f})")

        # 异常消费: 消费金额远超同等级用户
        if "total_spend" in self.df.columns and "user_level" in self.df.columns:
            for level in self.df["user_level"].unique():
                level_data = self.df[self.df["user_level"] == level]["total_spend"]
                upper = level_data.mean() + 3 * level_data.std()
                mask = (self.df["user_level"] == level) & (self.df["total_spend"] > upper)
                count = mask.sum()
                if count > 0:
                    self.df.loc[mask, "is_abnormal_spend"] = 1
                    flags.append(f"等级{level}异常消费: {count} 条 (金额>{upper:.0f})")

        self.df["is_abnormal_spend"] = self.df.get("is_abnormal_spend", pd.Series(0, index=self.df.index)).fillna(0).astype(int)

        self.clean_log.append(f"[异常值标记]")
        for f in flags:
            self.clean_log.append(f"  {f}")
        if not flags:
            self.clean_log.append(f"  未发现明显异常值")

    def _optimize_types(self):
        """优化数据类型，节省内存"""
        int_cols = ["age", "gender", "user_level", "purchase_freq", "register_days",
                    "follow_num", "fans_num", "img_count", "has_video", "like_num",
                    "comment_num", "share_num", "collect_num", "is_follow_author",
                    "add2cart", "coupon_received", "coupon_used", "pv_count", "label"]
        for col in int_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(int)

        self.clean_log.append(f"[类型优化] 整型列已转换")

    def _log_after(self):
        self.clean_log.append(f"\n[清洗完成] 最终数据: {len(self.df)} 行, {len(self.df.columns)} 列")
        self.clean_log.append(f"{'='*50}\n")

    # ── 查询接口 ──────────────────────────────────────────
    def query_sql(self, sql):
        """执行 SQL 查询"""
        conn = sqlite3.connect(self.db_path)
        result = pd.read_sql_query(sql, conn)
        conn.close()
        return result

    def get_clean_log(self):
        """返回清洗日志"""
        return "\n".join(self.clean_log)

    def get_summary(self):
        """返回数据概览"""
        if self.df is None:
            return {}
        return {
            "total_rows": len(self.df),
            "total_cols": len(self.df.columns),
            "total_users": self.df["user_id"].nunique() if "user_id" in self.df.columns else 0,
            "total_items": self.df["item_id"].nunique() if "item_id" in self.df.columns else 0,
            "overall_conversion": round(self.df["label"].mean() * 100, 2) if "label" in self.df.columns else 0,
            "total_gmv": round(self.df.loc[self.df["label"] == 1, "price"].sum(), 2) if "price" in self.df.columns else 0,
            "missing_values": int(self.df.isnull().sum().sum()),
            "anomaly_count": int(self.df.get("is_suspected_fraud", pd.Series(0)).sum()),
        }
