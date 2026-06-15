"""
消费倾向分析模块
- 逻辑回归 / 随机森林 / XGBoost 自动选优
- 输出购买概率 + 推荐等级 + 特征重要性
- 模型版本管理
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
    """购买意愿预测器"""

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
        self.y_test = self.y_test if hasattr(self, "y_test") else None
        self.version = datetime.now().strftime("%Y%m%d_%H%M%S")
        setup_style()

    def prepare_features(self):
        """特征工程"""
        # 选择特征列
        numeric_features = [
            "age", "user_level", "purchase_freq", "total_spend", "register_days",
            "follow_num", "fans_num", "price", "discount_rate", "title_length",
            "title_emo_score", "img_count", "has_video", "like_num", "comment_num",
            "share_num", "collect_num", "is_follow_author", "add2cart",
            "coupon_received", "coupon_used", "pv_count", "last_click_gap",
            "interaction_rate", "purchase_intent", "freshness_score", "social_influence"
        ]

        # 类别特征编码
        if "category" in self.df.columns:
            le = LabelEncoder()
            self.df["category_encoded"] = le.fit_transform(self.df["category"].fillna("未知"))
            self.label_encoders["category"] = le
            numeric_features.append("category_encoded")

        # 确保所有特征列存在
        self.feature_names = [f for f in numeric_features if f in self.df.columns]

        X = self.df[self.feature_names].fillna(0)
        y = self.df["label"].astype(int)

        # 标准化
        X_scaled = pd.DataFrame(self.scaler.fit_transform(X), columns=self.feature_names)

        # 划分数据集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y)

        return self.X_train, self.X_test, self.y_train, self.y_test

    def train_models(self):
        """训练多个模型并自动选优"""
        model_configs = {
            "逻辑回归": LogisticRegression(max_iter=1000, random_state=42),
            "随机森林": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
            "XGBoost": self._get_xgboost(),
        }

        results = {}
        for name, model in model_configs.items():
            if model is None:
                continue
            model.fit(self.X_train, self.y_train)
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

        # 选 F1 最高的模型
        if results:
            self.best_model_name = max(results, key=lambda k: results[k]["f1"])
            self.best_model = self.models[self.best_model_name]

        return results

    def _get_xgboost(self):
        """获取 XGBoost 模型（兼容未安装情况）"""
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

    def predict_single(self, user_data):
        """单用户预测"""
        user_df = pd.DataFrame([user_data])
        # 确保 category_encoded 存在
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

    def batch_predict(self, df_new=None):
        """批量预测"""
        target_df = df_new if df_new is not None else self.df
        features = target_df[self.feature_names].fillna(0)
        features_scaled = pd.DataFrame(self.scaler.transform(features), columns=self.feature_names)

        probabilities = self.best_model.predict_proba(features_scaled)[:, 1]
        levels = ["高意愿" if p >= 0.7 else ("中意愿" if p >= 0.4 else "低意愿") for p in probabilities]

        result = target_df[["user_id", "item_id"]].copy()
        result["purchase_probability"] = probabilities
        result["recommend_level"] = levels
        return result

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

        # 保存版本记录
        version_file = os.path.join(model_dir, "versions.json")
        versions = []
        if os.path.exists(version_file):
            with open(version_file, "r") as f:
                versions = json.load(f)
        versions.append({
            "version": self.version,
            "model_name": self.best_model_name,
            "path": model_path,
        })
        with open(version_file, "w") as f:
            json.dump(versions, f, indent=2)

        return model_path

    def get_prediction_stats(self, results):
        """返回预测统计"""
        top_features = {}
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
        }
        return stats

    def run(self):
        """运行完整预测流程"""
        self.prepare_features()
        results = self.train_models()
        stats = self.get_prediction_stats(results)

        charts = {
            "model_compare": self.plot_model_comparison(results),
            "feature_importance": self.plot_feature_importance(),
            "confusion_matrix": self.plot_confusion_matrix(),
        }

        model_path = self.save_model()
        stats["model_path"] = model_path

        return stats, charts
