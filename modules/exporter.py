"""
导出模块 - 生成结构清晰的 Excel 报告
"""
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from config import OUTPUT_DIR


class Exporter:
    """分析报告导出器"""

    def __init__(self):
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)

    def export_all(self, all_stats, all_charts):
        """导出所有分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"电商分析报告_{timestamp}.xlsx"
        filepath = os.path.join(self.output_dir, filename)

        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            # 总览 sheet
            self._export_summary(writer, all_stats)

            # 各模块详细数据
            if "dashboard" in all_stats:
                self._export_dashboard(writer, all_stats["dashboard"])
            if "user" in all_stats:
                self._export_user(writer, all_stats["user"])
            if "product" in all_stats:
                self._export_product(writer, all_stats["product"])
            if "coupon" in all_stats:
                self._export_coupon(writer, all_stats["coupon"])
            if "prediction" in all_stats:
                self._export_prediction(writer, all_stats["prediction"])
            if "social" in all_stats:
                self._export_social(writer, all_stats["social"])

        return filepath

    def _export_summary(self, writer, all_stats):
        """导出总览 sheet"""
        rows = []

        # 数据概览
        d = all_stats.get("dashboard", {})
        rows.append({"模块": "数据概览", "指标": "总用户数", "数值": d.get("total_users", "")})
        rows.append({"模块": "", "指标": "总商品数", "数值": d.get("total_items", "")})
        rows.append({"模块": "", "指标": "整体转化率", "数值": f"{d.get('overall_conversion', 0):.1f}%"})
        rows.append({"模块": "", "指标": "总GMV", "数值": f"¥{d.get('total_gmv', 0):,.2f}"})
        rows.append({"模块": "", "指标": "平均价格", "数值": f"¥{d.get('avg_price', 0):.2f}"})
        rows.append({"模块": "", "指标": "券使用率", "数值": f"{d.get('coupon_usage_rate', 0):.1f}%"})

        # 用户画像
        u = all_stats.get("user", {})
        rows.append({"模块": "用户画像", "指标": "平均年龄", "数值": f"{u.get('age_mean', 0):.1f} 岁"})
        g = u.get("gender_ratio", {})
        rows.append({"模块": "", "指标": "男女比例", "数值": f"男{g.get('男', 0)} / 女{g.get('女', 0)}"})
        rows.append({"模块": "", "指标": "平均消费", "数值": f"¥{u.get('avg_total_spend', 0):,.2f}"})

        # RFM 分层
        rfm = u.get("rfm_distribution", {})
        for lv, cnt in rfm.items():
            rows.append({"模块": "", "指标": lv, "数值": f"{cnt} 人"})

        # 商品分析
        p = all_stats.get("product", {})
        rows.append({"模块": "商品分析", "指标": "平均折扣率", "数值": f"{p.get('avg_discount', 0)*100:.1f}%"})
        v = p.get("video_impact", {})
        rows.append({"模块": "", "指标": "有视频购买率", "数值": f"{v.get('有视频购买率', 0):.1f}%"})
        rows.append({"模块": "", "指标": "无视频购买率", "数值": f"{v.get('无视频购买率', 0):.1f}%"})

        # 优惠券
        c = all_stats.get("coupon", {})
        rows.append({"模块": "优惠券策略", "指标": "券使用率", "数值": f"{c.get('usage_rate', 0):.1f}%"})
        rows.append({"模块": "", "指标": "最优折扣区间", "数值": c.get("best_discount_range", "")})
        rows.append({"模块": "", "指标": "最优区间转化率", "数值": f"{c.get('best_discount_rate', 0):.1f}%"})

        # 消费倾向
        pred = all_stats.get("prediction", {})
        rows.append({"模块": "消费倾向", "指标": "最优模型", "数值": pred.get("best_model", "")})

        # 社交影响力
        s = all_stats.get("social", {})
        rows.append({"模块": "社交影响力", "指标": "高影响力用户", "数值": f"{s.get('total_influencers', 0)} 人"})
        rows.append({"模块": "", "指标": "平均影响力评分", "数值": f"{s.get('avg_influence_score', 0):.1f}"})

        df = pd.DataFrame(rows)
        df.to_excel(writer, sheet_name="总览", index=False)

    def _export_dashboard(self, writer, stats):
        """导出数据概览"""
        rows = [
            {"指标": "总用户数", "数值": stats.get("total_users", "")},
            {"指标": "总商品数", "数值": stats.get("total_items", "")},
            {"指标": "总记录数", "数值": stats.get("total_records", "")},
            {"指标": "整体转化率", "数值": f"{stats.get('overall_conversion', 0):.1f}%"},
            {"指标": "总GMV", "数值": f"¥{stats.get('total_gmv', 0):,.2f}"},
            {"指标": "平均价格", "数值": f"¥{stats.get('avg_price', 0):.2f}"},
            {"指标": "平均浏览量", "数值": f"{stats.get('avg_pv', 0):.1f}"},
            {"指标": "券使用率", "数值": f"{stats.get('coupon_usage_rate', 0):.1f}%"},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="数据概览", index=False)

    def _export_user(self, writer, stats):
        """导出用户画像"""
        # 基础数据
        rows = [
            {"指标": "总用户数", "数值": stats.get("total_users", "")},
            {"指标": "平均年龄", "数值": f"{stats.get('age_mean', 0):.1f}"},
            {"指标": "年龄中位数", "数值": f"{stats.get('age_median', 0):.1f}"},
            {"指标": "平均购买频次", "数值": f"{stats.get('avg_purchase_freq', 0):.1f}"},
            {"指标": "平均消费金额", "数值": f"¥{stats.get('avg_total_spend', 0):,.2f}"},
            {"指标": "平均粉丝数", "数值": f"{stats.get('avg_fans', 0):.1f}"},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="用户画像", index=False)

        # RFM 分层
        rfm = stats.get("rfm_distribution", {})
        if rfm:
            rfm_rows = [{"分层": k, "用户数": v} for k, v in rfm.items()]
            pd.DataFrame(rfm_rows).to_excel(writer, sheet_name="RFM分层", index=False)

        # 价值分层
        vs = stats.get("value_segmentation", {})
        if vs:
            vs_rows = [{"分层": k, "用户数": v} for k, v in vs.items()]
            pd.DataFrame(vs_rows).to_excel(writer, sheet_name="价值分层", index=False)

    def _export_product(self, writer, stats):
        """导出商品分析"""
        rows = [
            {"指标": "平均价格", "数值": f"¥{stats.get('avg_price', 0):.2f}"},
            {"指标": "平均折扣率", "数值": f"{stats.get('avg_discount', 0)*100:.1f}%"},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="商品分析", index=False)

        # 类目统计
        cat = stats.get("category_stats", {})
        if cat and "purchase_rate" in cat:
            cat_rows = [{"类目": k, "购买率": f"{v*100:.1f}%"} for k, v in cat["purchase_rate"].items()]
            pd.DataFrame(cat_rows).to_excel(writer, sheet_name="类目购买率", index=False)

        # 热度 Top10
        hot = stats.get("top10_hot_items", [])
        if hot:
            pd.DataFrame(hot).to_excel(writer, sheet_name="热度Top10", index=False)

    def _export_coupon(self, writer, stats):
        """导出优惠券分析"""
        rows = [
            {"指标": "领券人数", "数值": stats.get("total_received", "")},
            {"指标": "用券人数", "数值": stats.get("total_used", "")},
            {"指标": "券使用率", "数值": f"{stats.get('usage_rate', 0):.1f}%"},
            {"指标": "未用券人数", "数值": stats.get("unused_user_count", "")},
            {"指标": "最优折扣区间", "数值": stats.get("best_discount_range", "")},
            {"指标": "最优区间转化率", "数值": f"{stats.get('best_discount_rate', 0):.1f}%"},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="优惠券分析", index=False)

        # 再触达建议
        sug = stats.get("retention_suggestions", [])
        if sug and isinstance(sug, list):
            pd.DataFrame({"建议": sug}).to_excel(writer, sheet_name="再触达建议", index=False)

    def _export_prediction(self, writer, stats):
        """导出消费倾向分析"""
        # 模型对比
        results = stats.get("model_results", {})
        if results:
            model_rows = []
            for name, m in results.items():
                model_rows.append({
                    "模型": name,
                    "准确率": f"{m.get('accuracy', 0):.3f}",
                    "精确率": f"{m.get('precision', 0):.3f}",
                    "召回率": f"{m.get('recall', 0):.3f}",
                    "F1": f"{m.get('f1', 0):.3f}",
                    "AUC": f"{m.get('auc', 0):.3f}",
                })
            pd.DataFrame(model_rows).to_excel(writer, sheet_name="模型对比", index=False)

        # 特征重要性
        feat = stats.get("top_features", {})
        if feat:
            feat_names = {
                "add2cart": "加购行为", "coupon_used": "使用优惠券",
                "is_follow_author": "关注作者", "interaction_rate": "互动率",
                "purchase_freq": "购买频次", "user_level": "用户等级",
                "pv_count": "浏览次数", "price": "商品价格",
                "discount_rate": "折扣力度", "fans_num": "粉丝数",
                "social_influence": "社交影响力", "freshness_score": "内容新鲜度",
                "total_spend": "历史消费", "register_days": "注册时长",
                "like_num": "点赞数", "comment_num": "评论数",
                "share_num": "分享数", "collect_num": "收藏数",
                "title_emo_score": "标题吸引力", "img_count": "图片数量",
                "has_video": "有无视频", "age": "年龄", "gender": "性别",
                "category_encoded": "商品类目", "title_length": "标题长度",
                "last_click_gap": "距上次点击", "purchase_intent": "购买意图",
            }
            feat_rows = [{"排名": i+1, "特征": feat_names.get(k, k), "重要性": f"{v:.4f}"}
                         for i, (k, v) in enumerate(feat.items())]
            pd.DataFrame(feat_rows).to_excel(writer, sheet_name="特征重要性", index=False)

    def _export_social(self, writer, stats):
        """导出社交影响力分析"""
        rows = [
            {"指标": "高影响力用户数", "数值": stats.get("total_influencers", "")},
            {"指标": "平均影响力评分", "数值": f"{stats.get('avg_influence_score', 0):.1f}"},
        ]
        pd.DataFrame(rows).to_excel(writer, sheet_name="社交影响力", index=False)

        # Top10 用户
        top = stats.get("top10_users", [])
        if top:
            top_rows = []
            for u in top:
                top_rows.append({
                    "用户ID": u.get("user_id", ""),
                    "影响力评分": f"{u.get('influence_score', 0):.1f}",
                    "粉丝数": u.get("fans_num", ""),
                    "互动率": f"{u.get('avg_interaction_rate', 0):.1f}",
                    "购买率": f"{u.get('purchase_rate', 0)*100:.1f}%",
                })
            pd.DataFrame(top_rows).to_excel(writer, sheet_name="Top10用户", index=False)

        # 影响力分布
        sid = stats.get("social_influence_distribution", {})
        if sid:
            sid_rows = [{"分层": k, "商品数": v} for k, v in sid.items()]
            pd.DataFrame(sid_rows).to_excel(writer, sheet_name="影响力分布", index=False)
