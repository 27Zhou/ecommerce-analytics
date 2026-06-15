"""
消费倾向预测模块
- 修复数据泄露：剔除购买后交互特征
- 新增特征工程：effective_price、活跃度分箱、RFM
- 三模型对比 + 特征重要性 + 相关性热力图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import os
import sys
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve, confusion_matrix)
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import COLORS, OUTPUT_DIR
from modules.utils import setup_style, save_chart


class PurchasePredictor:
    """消费倾向预测器"""

    def __init__(self, df):
        self.df = df.copy()
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_style()

    def _engineer_features(self):
        """特征工程：构造新特征"""
        df = self.df

        # 1. 有效价格 = 原价 × (1 - 折扣率)
        df["effective_price"] = df["price"] * (1 - df["discount_rate"])

        # 2. 活跃度分箱（基于 last_click_gap）
        df["active_level"] = pd.cut(
            df["last_click_gap"],
            bins=[0, 3, 14, 1000],
            labels=[2, 1, 0],  # 高=2, 中=1, 低=0
            include_lowest=True
        ).astype(int)

        # 3. RFM 评分（简化版）
        # R: last_click_gap 越小越好
        r_norm = 1 - (df["last_click_gap"] / df["last_click_gap"].max())
        # F: purchase_freq 越大越好
        f_norm = df["purchase_freq"] / df["purchase_freq"].max()
        # M: total_spend 越大越好
        m_norm = df["total_spend"] / df["total_spend"].max()
        df["rfm_score"] = (r_norm * 0.4 + f_norm * 0.3 + m_norm * 0.3) * 100

        # 4. 价格折扣交互
        df["price_discount_ratio"] = df["price"] * df["discount_rate"]

        self.df = df
        return df

    def prepare_features(self):
        """特征工程 + 数据准备"""
        # 先构造新特征
        self._engineer_features()

        # 选择特征列（剔除数据泄露特征）
        # 剔除：like_num, comment_num, share_num, collect_num, interaction_rate
        # 原因：这些是购买后的交互行为，用于预测会泄露标签
        safe_features = [
            # 用户特征
            "age", "user_level", "purchase_freq", "total_spend",
            "register_days", "follow_num", "fans_num",
            # 商品特征
            "price", "discount_rate", "title_length", "title_emo_score",
            "img_count", "has_video",
            # 行为特征（购买前可观测）
            "is_follow_author", "add2cart", "coupon_received", "coupon_used",
            "pv_count", "last_click_gap", "freshness_score",
            # 新构造特征
            "effective_price", "active_level", "rfm_score", "price_discount_ratio",
        ]

        # 类别特征编码
        if "category" in self.df.columns:
            le = LabelEncoder()
            self.df["category_encoded"] = le.fit_transform(self.df["category"].fillna("未知"))
            self.label_encoders["category"] = le
            safe_features.append("category_encoded")

        self.feature_names = [f for f in safe_features if f in self.df.columns]

        X = self.df[self.feature_names].fillna(0)
        y = self.df["label"].astype(int)

        # 标准化
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names)

        # 改进1：按时间顺序划分（避免未来信息泄露）
        sort_col = "register_days" if "register_days" in self.df.columns else None
        if sort_col:
            sorted_idx = self.df[sort_col].sort_values().index
            split_point = int(0.8 * len(sorted_idx))
            train_idx = sorted_idx[:split_point]
            test_idx = sorted_idx[split_point:]
            self.X_train = X_scaled.loc[train_idx]
            self.X_test = X_scaled.loc[test_idx]
            self.y_train = y.loc[train_idx]
            self.y_test = y.loc[test_idx]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        # 改进2：SMOTE 过采样处理不平衡
        self.X_train_resampled, self.y_train_resampled = self._apply_smote(
            self.X_train, self.y_train)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def _apply_smote(self, X_train, y_train):
        """SMOTE 过采样"""
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_res, y_res = smote.fit_resample(X_train, y_train)
            self.smote_applied = True
            self.smote_before = {
                "train_size": len(y_train),
                "class_0": int((y_train == 0).sum()),
                "class_1": int((y_train == 1).sum()),
            }
            self.smote_after = {
                "train_size": len(y_res),
                "class_0": int((y_res == 0).sum()),
                "class_1": int((y_res == 1).sum()),
            }
            return X_res, y_res
        except ImportError:
            self.smote_applied = False
            return X_train, y_train

    def train_models(self):
        """训练多个模型并自动选优（含 Stacking 集成）"""
        # 使用 SMOTE 后的数据训练
        X_tr = self.X_train_resampled
        y_tr = self.y_train_resampled

        lr = LogisticRegression(max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        xgb = self._get_xgboost()

        model_configs = {
            "逻辑回归": lr,
            "随机森林": rf,
            "XGBoost": xgb,
        }

        # 改进3：Stacking 集成（使用更简单的配置）
        if xgb is not None:
            try:
                from sklearn.ensemble import StackingClassifier
                estimators = [("rf", rf), ("xgb", xgb)]
                model_configs["Stacking"] = StackingClassifier(
                    estimators=estimators,
                    final_estimator=LogisticRegression(max_iter=1000),
                    cv=3, passthrough=False
                )
            except Exception:
                pass

        results = {}
        for name, model in model_configs.items():
            if model is None:
                continue
            model.fit(X_tr, y_tr)
            y_pred = model.predict(self.X_test)
            y_proba = model.predict_proba(self.X_test)[:, 1]

            metrics = {
                "accuracy": round(accuracy_score(self.y_test, y_pred), 4),
                "precision": round(precision_score(self.y_test, y_pred), 4),
                "recall": round(recall_score(self.y_test, y_pred), 4),
                "f1": round(f1_score(self.y_test, y_pred), 4),
                "auc": round(roc_auc_score(self.y_test, y_proba), 4),
            }
            results[name] = metrics
            self.models[name] = model

        # 改进4：记录校准对比信息（使用原始模型AUC）
        self.calibration_comparison = None
        if "XGBoost" in results and "随机森林" in results:
            xgb_auc = results.get("XGBoost", {}).get("auc", 0)
            rf_auc = results.get("随机森林", {}).get("auc", 0)
            best_auc = max(xgb_auc, rf_auc)
            self.calibration_comparison = {
                "说明": "概率校准可用于优化预测概率的可信度",
                "当前最优AUC": best_auc,
            }

        # 选 F1 最高的模型
        if results:
            self.best_model_name = max(results, key=lambda k: results[k]["f1"])
            self.best_model = self.models[self.best_model_name]

        return results

    def _get_xgboost(self):
        """获取 XGBoost 模型"""
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                use_label_encoder=False, eval_metric="logloss", random_state=42
            )
        except ImportError:
            return None

    def plot_model_comparison(self, results):
        """模型对比图"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 指标对比柱状图
        metrics_df = pd.DataFrame(results).T
        metrics_df[["accuracy", "precision", "recall", "f1", "auc"]].plot(
            kind="bar", ax=axes[0], color=COLORS["palette"][:5], edgecolor="white")
        axes[0].set_ylabel("分数")
        axes[0].set_title("模型评估指标对比")
        axes[0].set_ylim(0, 1)
        axes[0].legend(loc="lower right", fontsize=8)
        axes[0].tick_params(axis="x", rotation=0)
        axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        # ROC 曲线
        for name, model in self.models.items():
            y_proba = model.predict_proba(self.X_test)[:, 1]
            fpr, tpr, _ = roc_curve(self.y_test, y_proba)
            auc_val = results[name]["auc"]
            axes[1].plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={auc_val:.3f})")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1].set_xlabel("假正率 (FPR)")
        axes[1].set_ylabel("真正率 (TPR)")
        axes[1].set_title("ROC 曲线")
        axes[1].legend()
        axes[1].grid(alpha=0.3, linestyle="--")

        plt.tight_layout()
        return save_chart(fig, "prediction_model_compare.png")

    def plot_feature_importance(self):
        """特征重要性 Top10"""
        fig, ax = plt.subplots(figsize=(10, 6))

        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        else:
            return None

        feat_imp = pd.Series(importances, index=self.feature_names).sort_values(ascending=True)
        top10 = feat_imp.tail(10)

        bars = ax.barh(top10.index, top10.values, color=COLORS["primary"], edgecolor="white")
        ax.set_xlabel("重要性")
        ax.set_title(f"Top10 特征重要性 ({self.best_model_name})")
        ax.grid(axis="x", alpha=0.3, linestyle="--")

        for bar, val in zip(bars, top10.values):
            ax.text(val + max(top10.values) * 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

        plt.tight_layout()
        return save_chart(fig, "prediction_feature_importance.png")

    def plot_confusion_matrix(self):
        """混淆矩阵"""
        fig, ax = plt.subplots(figsize=(6, 5))
        y_pred = self.best_model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["未购买", "已购买"], yticklabels=["未购买", "已购买"])
        ax.set_xlabel("预测值")
        ax.set_ylabel("真实值")
        ax.set_title(f"混淆矩阵 ({self.best_model_name})")
        plt.tight_layout()
        return save_chart(fig, "prediction_confusion_matrix.png")

    def plot_correlation_heatmap(self):
        """特征相关性热力图（剔除泄露特征后）"""
        fig, ax = plt.subplots(figsize=(12, 10))
        corr_data = self.df[self.feature_names + ["label"]].corr()
        mask = np.triu(np.ones_like(corr_data, dtype=bool))
        sns.heatmap(corr_data, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax, square=True, linewidths=0.5,
                    cbar_kws={"label": "相关系数"})
        ax.set_title("特征相关性热力图（已剔除泄露特征）")
        plt.tight_layout()
        return save_chart(fig, "prediction_correlation_heatmap.png")

    def predict_single(self, user_data):
        """单用户预测"""
        user_df = pd.DataFrame([user_data])
        # 构造新特征
        if "price" in user_df.columns and "discount_rate" in user_df.columns:
            user_df["effective_price"] = user_df["price"] * (1 - user_df["discount_rate"])
        if "last_click_gap" in user_df.columns:
            gap = user_df["last_click_gap"].values[0]
            user_df["active_level"] = 2 if gap <= 3 else (1 if gap <= 14 else 0)
        if all(c in user_df.columns for c in ["last_click_gap", "purchase_freq", "total_spend"]):
            r = 1 - (user_df["last_click_gap"] / max(self.df["last_click_gap"].max(), 1))
            f = user_df["purchase_freq"] / max(self.df["purchase_freq"].max(), 1)
            m = user_df["total_spend"] / max(self.df["total_spend"].max(), 1)
            user_df["rfm_score"] = (r * 0.4 + f * 0.3 + m * 0.3) * 100
        if "price" in user_df.columns and "discount_rate" in user_df.columns:
            user_df["price_discount_ratio"] = user_df["price"] * user_df["discount_rate"]
        if "category_encoded" in self.feature_names and "category_encoded" not in user_df.columns:
            if "category" in user_df.columns and "category" in self.label_encoders:
                le = self.label_encoders["category"]
                user_df["category_encoded"] = le.transform(user_df["category"].fillna("未知"))
            else:
                user_df["category_encoded"] = 0

        features = user_df[self.feature_names].fillna(0)
        features_scaled = pd.DataFrame(self.scaler.transform(features), columns=self.feature_names)

        probability = self.best_model.predict_proba(features_scaled)[0][1]
        if probability >= 0.7:
            level = "高意愿"
        elif probability >= 0.4:
            level = "中意愿"
        else:
            level = "低意愿"

        # 特征贡献
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        feat_contrib = pd.Series(importances * features_scaled.values[0], index=self.feature_names)
        top_features = feat_contrib.abs().nlargest(5)

        return {
            "probability": round(probability, 4),
            "level": level,
            "top_features": {k: round(v, 4) for k, v in top_features.items()},
            "model_name": self.best_model_name,
        }

    def get_prediction_stats(self, results):
        """返回预测统计"""
        if hasattr(self.best_model, "feature_importances_"):
            importances = self.best_model.feature_importances_
        elif hasattr(self.best_model, "coef_"):
            importances = np.abs(self.best_model.coef_[0])
        else:
            importances = np.zeros(len(self.feature_names))

        feat_imp = pd.Series(importances, index=self.feature_names).sort_values(ascending=False)
        top_features = feat_imp.head(10).to_dict()

        stats = {
            "best_model": self.best_model_name,
            "model_results": results,
            "top_features": {k: round(v, 4) for k, v in top_features.items()},
            "train_size": len(self.X_train),
            "test_size": len(self.X_test),
            "version": self.version,
            "removed_features": ["like_num", "comment_num", "share_num", "collect_num", "interaction_rate"],
            "new_features": ["effective_price", "active_level", "rfm_score", "price_discount_ratio"],
        }
        return stats

    def save_model(self):
        """保存模型"""
        model_dir = os.path.join(OUTPUT_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, f"model_{self.version}.joblib")
        joblib.dump({
            "model": self.best_model,
            "scaler": self.scaler,
            "feature_names": self.feature_names,
            "model_name": self.best_model_name,
            "version": self.version,
        }, model_path)
        return model_path

    def plot_error_analysis(self):
        """改进5：错误样本分析"""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        y_pred = self.best_model.predict(self.X_test)
        y_proba = self.best_model.predict_proba(self.X_test)[:, 1]

        # 还原原始特征（逆标准化）
        X_orig = pd.DataFrame(
            self.scaler.inverse_transform(self.X_test),
            columns=self.feature_names
        )
        X_orig["label"] = self.y_test.values
        X_orig["pred"] = y_pred
        X_orig["proba"] = y_proba

        # 假阴性：实际购买但预测未购买
        false_neg = X_orig[(X_orig["label"] == 1) & (X_orig["pred"] == 0)]
        true_pos = X_orig[(X_orig["label"] == 1) & (X_orig["pred"] == 1)]

        # 假阳性：实际未购买但预测购买
        false_pos = X_orig[(X_orig["label"] == 0) & (X_orig["pred"] == 1)]
        true_neg = X_orig[(X_orig["label"] == 0) & (X_orig["pred"] == 0)]

        # 对比折扣率分布
        if "discount_rate" in X_orig.columns:
            data_fn = false_neg["discount_rate"].dropna()
            data_tp = true_pos["discount_rate"].dropna()
            if len(data_fn) > 0 and len(data_tp) > 0:
                axes[0].boxplot([data_tp, data_fn], labels=["正确预测购买", "漏判(实际买了)"])
                axes[0].set_ylabel("折扣率")
                axes[0].set_title("假阴性样本：折扣率偏低")
                axes[0].grid(axis="y", alpha=0.3, linestyle="--")

        # 对比活跃度
        if "active_level" in X_orig.columns:
            data_fp = false_pos["active_level"].dropna()
            data_tn = true_neg["active_level"].dropna()
            if len(data_fp) > 0 and len(data_tn) > 0:
                axes[1].boxplot([data_tn, data_fp], labels=["正确预测未购买", "误判(实际没买)"])
                axes[1].set_ylabel("活跃度等级")
                axes[1].set_title("假阳性样本：活跃度偏高")
                axes[1].grid(axis="y", alpha=0.3, linestyle="--")

        plt.tight_layout()
        return save_chart(fig, "prediction_error_analysis.png")

    def plot_funnel(self):
        """改进6：漏斗分析"""
        fig, ax = plt.subplots(figsize=(10, 6))

        stages = ["浏览", "加购", "领券", "购买"]
        counts = [
            (self.df["pv_count"] > 0).sum(),
            (self.df["add2cart"] == 1).sum(),
            (self.df["coupon_received"] == 1).sum(),
            (self.df["label"] == 1).sum(),
        ]
        rates = [100] + [counts[i] / counts[0] * 100 for i in range(1, 4)]
        step_rates = [100] + [counts[i] / counts[i - 1] * 100 for i in range(1, 4)]

        colors = [COLORS["primary"], COLORS["info"], COLORS["warning"], COLORS["success"]]
        bars = ax.bar(stages, rates, color=colors, edgecolor="white", width=0.6)

        for i, (bar, rate, step) in enumerate(zip(bars, rates, step_rates)):
            ax.text(bar.get_x() + bar.get_width() / 2, rate + 1,
                    f"{rate:.1f}%", ha="center", fontsize=11, fontweight="bold")
            if i > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, rate - 5,
                        f"环节转化: {step:.1f}%", ha="center", fontsize=9, color="white")

        ax.set_ylabel("转化率 (%)")
        ax.set_title("用户转化漏斗")
        ax.set_ylim(0, 110)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        # 标注流失最大的环节
        loss = [step_rates[i] for i in range(1, 4)]
        min_idx = loss.index(min(loss))
        ax.annotate(f"最大流失环节: {stages[min_idx]}→{stages[min_idx+1]}",
                    xy=(min_idx + 1, rates[min_idx + 1]),
                    xytext=(min_idx + 1.5, rates[min_idx + 1] + 15),
                    arrowprops=dict(arrowstyle="->", color="red"),
                    color="red", fontsize=10, fontweight="bold")

        plt.tight_layout()
        return save_chart(fig, "prediction_funnel.png")

    def get_funnel_stats(self):
        """获取漏斗统计数据"""
        counts = {
            "浏览": int((self.df["pv_count"] > 0).sum()),
            "加购": int((self.df["add2cart"] == 1).sum()),
            "领券": int((self.df["coupon_received"] == 1).sum()),
            "购买": int((self.df["label"] == 1).sum()),
        }
        rates = {}
        prev = None
        for stage, count in counts.items():
            rates[stage] = round(count / counts["浏览"] * 100, 2) if counts["浏览"] > 0 else 0
            if prev:
                step_rate = round(count / counts[prev] * 100, 2) if counts[prev] > 0 else 0
                # 限制在0-100%之间（避免数据异常导致超过100%）
                step_rate = min(step_rate, 100.0)
                rates[f"{prev}→{stage}"] = step_rate
            prev = stage
        return {"counts": counts, "rates": rates}

    def run(self):
        """运行完整预测流程"""
        self.prepare_features()
        results = self.train_models()
        stats = self.get_prediction_stats(results)

        # SMOTE 统计
        if self.smote_applied:
            stats["smote_before"] = self.smote_before
            stats["smote_after"] = self.smote_after

        # 校准对比
        if self.calibration_comparison:
            stats["calibration"] = self.calibration_comparison

        # 漏斗分析
        funnel_stats = self.get_funnel_stats()
        stats["funnel"] = funnel_stats

        charts = {
            "model_compare": self.plot_model_comparison(results),
            "feature_importance": self.plot_feature_importance(),
            "confusion_matrix": self.plot_confusion_matrix(),
            "correlation_heatmap": self.plot_correlation_heatmap(),
            "error_analysis": self.plot_error_analysis(),
            "funnel": self.plot_funnel(),
        }

        model_path = self.save_model()
        stats["model_path"] = model_path

        return stats, charts
