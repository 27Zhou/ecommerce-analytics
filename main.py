"""
社交电商数据分析系统 - 主程序
"""
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import os
import sys
import subprocess
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_DIR, OUTPUT_DIR
from modules.data_loader import DataLoader
from modules.dashboard import DashboardAnalyzer
from modules.user_analysis import UserAnalyzer
from modules.product_analysis import ProductAnalyzer
from modules.coupon_analysis import CouponAnalyzer
from modules.prediction import PurchasePredictor
from modules.social_analysis import SocialAnalyzer
from modules.exporter import Exporter
from PIL import Image, ImageTk


class ChartViewer:
    """图表查看器"""

    def __init__(self, parent):
        self.frame = ttk.Frame(parent)
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.charts = {}
        self.chart_list = []
        self._photo = None
        self._current_path = None

        top = ttk.Frame(self.frame)
        top.pack(fill=tk.X, padx=5, pady=2)
        ttk.Label(top, text="图表:").pack(side=tk.LEFT, padx=(0, 5))
        self.combo = ttk.Combobox(top, state="readonly", width=25)
        self.combo.pack(side=tk.LEFT, padx=2)
        self.combo.bind("<<ComboboxSelected>>", self._on_combo)
        ttk.Button(top, text="◀", width=3, command=self._prev).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="▶", width=3, command=self._next).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text="🔍 放大", command=self._popup_view).pack(side=tk.LEFT, padx=8)
        self.idx_label = ttk.Label(top, text="")
        self.idx_label.pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Label(self.frame, bg="#F5F5F5", anchor="center", cursor="hand2")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.canvas.bind("<Double-Button-1>", self._on_double_click)

    def _on_combo(self, _):
        self._show(self.combo.current())

    def _on_double_click(self, _):
        self._popup_view()

    def _popup_view(self):
        """弹窗查看原图，自适应缩放"""
        if not self._current_path or not os.path.exists(self._current_path):
            return

        win = tk.Toplevel(self.frame)
        win.title("图表查看")
        win.configure(bg="white")

        # 获取屏幕尺寸
        sw = win.winfo_screenwidth()
        sh = win.winfo_screenheight()
        win.geometry(f"{sw-100}x{sh-100}+50+50")

        img = Image.open(self._current_path)
        # 窗口打开后再缩放
        win.update_idletasks()
        ww = max(sw - 120, 800)
        wh = max(sh - 180, 600)
        scale = min(ww / img.width, wh / img.height, 1.0)
        new_w, new_h = int(img.width * scale), int(img.height * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)

        photo = ImageTk.PhotoImage(img)
        label = tk.Label(win, image=photo, bg="white")
        label.image = photo
        label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        tip = ttk.Label(win, text="按 Esc 或关闭窗口退出", foreground="gray")
        tip.pack(pady=(0, 5))
        win.bind("<Escape>", lambda e: win.destroy())

    def load_charts(self, charts_dict):
        self.charts = charts_dict
        self.chart_list = list(charts_dict.keys())
        if self.chart_list:
            self.combo["values"] = [self._pretty(n) for n in self.chart_list]
            self.combo.current(0)
            self._show(0)
        else:
            self.combo["values"] = []
            self.canvas.config(image="", text="暂无图表")
            self._current_path = None

    def _pretty(self, n):
        m = {"pv_trend": "浏览量分布", "category_sales": "类目销售", "user_level": "用户等级",
             "age_dist": "年龄分布", "gender_ratio": "性别比例", "rfm": "RFM分层",
             "social_dist": "社交分布", "category_heatmap": "类目热度", "price_conversion": "价格转化率",
             "discount_curve": "折扣曲线", "content_impact": "内容影响", "title_sentiment": "标题情感",
             "usage_rate": "券使用率", "ab_matrix": "AB测试", "unused_users": "未用券用户",
             "model_compare": "模型对比", "feature_importance": "特征重要性", "confusion_matrix": "混淆矩阵",
             "fan_dist": "粉丝分布", "top20_interaction": "互动Top20", "influence_dist": "影响力分布",
             "category_preference": "类目偏好", "hotness": "商品热度指数",
             "value_segmentation": "用户价值分层", "social_influence_index": "社交影响力指数"}
        return m.get(n, n)

    def _show(self, idx):
        if idx < 0 or idx >= len(self.chart_list):
            return
        path = self.charts[self.chart_list[idx]]
        self._current_path = path
        self.idx_label.config(text=f"{idx+1}/{len(self.chart_list)}")
        if not os.path.exists(path):
            self.canvas.config(image="", text="文件不存在")
            return
        try:
            img = Image.open(path)
            self.frame.update_idletasks()
            mw = max(self.canvas.winfo_width() - 20, 600)
            mh = max(self.canvas.winfo_height() - 20, 400)
            scale = min(mw / img.width, mh / img.height, 1.0)
            new_w, new_h = int(img.width * scale), int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self.canvas.config(image=self._photo, text="")
        except Exception as e:
            self.canvas.config(image="", text=str(e))

    def _prev(self):
        if self.chart_list:
            c = self.combo.current()
            self.combo.current(max(0, c - 1))
            self._show(self.combo.current())

    def _next(self):
        if self.chart_list:
            c = self.combo.current()
            self.combo.current(min(len(self.chart_list) - 1, c + 1))
            self._show(self.combo.current())


class App:
    """主应用"""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("社交电商数据分析系统")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)

        self.loader = DataLoader()
        self.exporter = Exporter()
        self.all_stats = {}
        self.all_charts = {}
        self.module_charts = {}

        self._build_ui()
        self._auto_detect_csv()

    def run(self):
        self.root.mainloop()

    def _build_ui(self):
        # 工具栏
        bar = ttk.Frame(self.root)
        bar.pack(fill=tk.X, padx=10, pady=5)
        ttk.Button(bar, text="📂 选择文件", command=self._select_file).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="🧹 数据清洗", command=self._clean_data).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="📊 一键分析", command=self._run_all).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="💾 导出报告", command=self._export).pack(side=tk.LEFT, padx=3)
        ttk.Button(bar, text="📁 打开目录", command=self._open_output).pack(side=tk.LEFT, padx=3)
        self.status_var = tk.StringVar(value="就绪")
        ttk.Label(bar, textvariable=self.status_var, foreground="gray").pack(side=tk.RIGHT, padx=10)

        # 主区域: 左导航 + 右内容
        main = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # 左侧导航
        nav = ttk.LabelFrame(main, text="分析模块", width=150)
        main.add(nav, weight=0)
        for text, cmd in [("📊 数据概览", self._show_dashboard), ("👤 用户画像", self._show_user),
                          ("📦 商品分析", self._show_product), ("🎫 优惠券", self._show_coupon),
                          ("🔮 消费倾向分析", self._show_prediction), ("📢 社交影响力", self._show_social)]:
            ttk.Button(nav, text=text, command=cmd, width=18).pack(fill=tk.X, padx=3, pady=2)

        # 右侧内容区
        right = ttk.Frame(main)
        main.add(right, weight=1)

        # 使用 Notebook 标签页切换图表和分析
        self.notebook = ttk.Notebook(right)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab1: 图表展示
        chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(chart_tab, text="  📊 图表展示  ")
        self.chart_viewer = ChartViewer(chart_tab)

        # Tab2: 分析说明
        info_tab = ttk.Frame(self.notebook)
        self.notebook.add(info_tab, text="  📋 分析说明  ")
        self.info = tk.Text(info_tab, wrap=tk.WORD, font=("Microsoft YaHei", 11), padx=12, pady=8,
                            spacing1=3, spacing2=2, spacing3=3, state=tk.DISABLED)
        sb = ttk.Scrollbar(info_tab, command=self.info.yview)
        self.info.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.info.pack(fill=tk.BOTH, expand=True)

        self._setup_tags()
        self._show_welcome()

    def _setup_tags(self):
        self.info.tag_configure("title", font=("Microsoft YaHei", 14, "bold"), foreground="#2C3E50")
        self.info.tag_configure("sec", font=("Microsoft YaHei", 11, "bold"), foreground="#2980B9", spacing1=8)
        self.info.tag_configure("val", font=("Consolas", 11, "bold"), foreground="#2C3E50")
        self.info.tag_configure("good", foreground="#27AE60")
        self.info.tag_configure("warn", foreground="#F39C12")
        self.info.tag_configure("bad", foreground="#E74C3C")
        self.info.tag_configure("tip", foreground="#8E44AD", font=("Microsoft YaHei", 11, "bold"))
        self.info.tag_configure("dim", foreground="#95A5A6", font=("Microsoft YaHei", 10))
        self.info.tag_configure("explain", foreground="#555555", font=("Microsoft YaHei", 10))

    def _clear_info(self):
        """清空分析区"""
        self.info.config(state=tk.NORMAL)
        self.info.delete(1.0, tk.END)
        self.info.config(state=tk.DISABLED)

    def _w(self, t, tag="normal"):
        self.info.config(state=tk.NORMAL)
        self.info.insert(tk.END, t, tag)
        self.info.config(state=tk.DISABLED)

    def _wl(self, t="", tag="normal"):
        self._w(t + "\n", tag)

    def _metric(self, label, value):
        self._w(f"  {label}:  ", "normal")
        self._wl(value, "val")

    def _hr(self):
        self._wl("─" * 45, "dim")

    def _show_welcome(self):
        self._clear_info()
        self._wl("社交电商数据分析系统 v2.0", "title")
        self._wl()

        self._wl("系统架构", "sec")
        self._wl("  ┌─────────────────────────────────────────┐")
        self._wl("  │         用户界面层 (Tkinter GUI)         │")
        self._wl("  ├─────────────────────────────────────────┤")
        self._wl("  │         业务逻辑层 (Python)              │")
        self._wl("  │    Pandas + Matplotlib + Scikit-learn    │")
        self._wl("  ├─────────────────────────────────────────┤")
        self._wl("  │         数据层 (CSV + SQLite)            │")
        self._wl("  └─────────────────────────────────────────┘")
        self._wl()

        self._wl("数据流程", "sec")
        self._wl("  CSV文件 → 数据导入 → 数据清洗 → 统计分析 → 可视化 → 报告导出")
        self._wl()

        self._wl("功能模块", "sec")
        self._wl("  ┌──────┬──────┬──────┬──────┬──────┬──────┐")
        self._wl("  │数据  │用户  │商品  │优惠券│消费  │社交  │")
        self._wl("  │概览  │画像  │分析  │策略  │倾向  │影响力│")
        self._wl("  └──────┴──────┴──────┴──────┴──────┴──────┘")
        self._wl()

        self._wl("使用说明", "sec")
        self._wl("  1. 将 CSV 放入 data 文件夹，或点击「选择文件」")
        self._wl("  2. 点击「数据清洗」→「一键分析」")
        self._wl("  3. 左侧选择模块，切换标签页查看图表或分析")
        self._wl("  4. 双击图表或点击「放大」可全屏查看")
        self._wl("  5. 点击「导出报告」生成 Excel")

    def _auto_detect_csv(self):
        csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
        if csv_files:
            self._load_file(os.path.join(DATA_DIR, csv_files[0]))

    def _select_file(self):
        fp = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")], initialdir=DATA_DIR)
        if fp:
            self._load_file(fp)

    def _load_file(self, fp):
        try:
            self.loader.load_csv(fp)
            self.status_var.set(f"已加载 {len(self.loader.df)} 条")
            self.notebook.select(1)  # 切换到分析标签
            self._clear_info()
            self._wl("✅ 数据加载成功", "title")
            self._wl()
            self._metric("文件", os.path.basename(fp))
            self._metric("记录数", f"{len(self.loader.df):,}")
        except Exception as e:
            messagebox.showerror("失败", str(e))

    def _clean_data(self):
        if self.loader.df is None:
            messagebox.showwarning("提示", "请先加载数据")
            return
        self.loader.clean()
        self.loader.load_to_sqlite()
        self.notebook.select(1)
        self._clear_info()
        self._wl("🧹 清洗完成", "title")
        self._wl()
        self._wl(self.loader.get_clean_log())

    def _run_all(self):
        if self.loader.df is None:
            messagebox.showwarning("提示", "请先加载数据")
            return
        self.status_var.set("分析中...")
        threading.Thread(target=self._run_all_thread, daemon=True).start()

    def _run_all_thread(self):
        try:
            modules = [
                ("dashboard", "数据概览", DashboardAnalyzer),
                ("user", "用户画像", UserAnalyzer),
                ("product", "商品分析", ProductAnalyzer),
                ("coupon", "优惠券策略", CouponAnalyzer),
                ("prediction", "消费倾向分析", PurchasePredictor),
                ("social", "社交影响力", SocialAnalyzer),
            ]
            total_charts = 0
            for name, _, cls in modules:
                stats, charts = cls(self.loader.df).run()
                self.all_stats[name] = stats
                self.all_charts.update(charts)
                self.module_charts[name] = charts
                total_charts += len(charts)

            self.status_var.set("分析完成")
            self.root.after(0, lambda: self._show_analysis_summary(total_charts))
        except Exception as e:
            self.status_var.set("出错")
            self.root.after(0, lambda: messagebox.showerror("出错", str(e)))

    def _show_analysis_summary(self, total_charts):
        """显示分析完成汇总"""
        self.notebook.select(1)
        self._clear_info()
        self._wl("✅ 全部分析完成", "title")
        self._wl()
        self._wl("已运行模块", "sec")
        self._wl("  ✓ 数据概览 — 核心指标、浏览量、类目销售、用户等级")
        self._wl("  ✓ 用户画像 — 年龄、性别、RFM分层、价值分层")
        self._wl("  ✓ 商品分析 — 类目热度、价格区间、折扣曲线、热度指数")
        self._wl("  ✓ 优惠券策略 — 使用率、A/B测试、未用券用户")
        self._wl("  ✓ 消费倾向 — 预测模型、特征重要性")
        self._wl("  ✓ 社交影响力 — 粉丝分布、KOL排行、影响力指数")
        self._wl()
        self._wl(f"共生成 {total_charts} 张图表", "val")
        self._wl()
        self._wl("操作提示", "tip")
        self._wl("  → 点击左侧模块查看详细分析")
        self._wl("  → 切换标签页查看图表")
        self._wl("  → 点击「导出报告」生成 Excel")

    def _ensure(self, name, cls):
        if name not in self.all_stats:
            if self.loader.df is None:
                messagebox.showwarning("提示", "请先加载数据")
                return False
            stats, charts = cls(self.loader.df).run()
            self.all_stats[name] = stats
            self.all_charts.update(charts)
            self.module_charts[name] = charts
        return True

    def _render(self, name, fn):
        cls = {"dashboard": DashboardAnalyzer, "user": UserAnalyzer, "product": ProductAnalyzer,
               "coupon": CouponAnalyzer, "prediction": PurchasePredictor, "social": SocialAnalyzer}[name]
        if not self._ensure(name, cls):
            return
        self.chart_viewer.load_charts(self.module_charts.get(name, {}))
        self.notebook.select(1)  # 默认显示分析标签
        self._clear_info()
        fn(self.all_stats.get(name, {}))

    # ── 数据概览 ──────────────────────────────────────────
    def _show_dashboard(self):
        def go(s):
            self._wl("📊 数据概览", "title"); self._hr(); self._wl()
            self._wl("核心指标", "sec")
            self._metric("总用户数", f"{s.get('total_users',0):,} 人")
            self._metric("总商品数", f"{s.get('total_items',0):,} 个")
            self._metric("整体转化率", f"{s.get('overall_conversion',0):.1f}%")
            self._metric("总 GMV", f"¥{s.get('total_gmv',0):,.2f}")
            self._metric("平均价格", f"¥{s.get('avg_price',0):.2f}")
            self._metric("平均浏览量", f"{s.get('avg_pv',0):.1f} 次")
            self._metric("券使用率", f"{s.get('coupon_usage_rate',0):.1f}%")
            self._wl()

            c = s.get('overall_conversion', 0)
            pv = s.get('avg_pv', 0)
            cr = s.get('coupon_usage_rate', 0)

            self._wl("分析结论", "sec"); self._hr()
            if c > 15:
                self._wl(f"  ✅ 转化率 {c:.1f}% — 表现优秀", "good")
                self._wl("     商品吸引力强，用户购买意愿高", "explain")
            elif c > 10:
                self._wl(f"  ⚠️ 转化率 {c:.1f}% — 中等水平", "warn")
                self._wl("     有提升空间，建议优化商品展示和营销策略", "explain")
            else:
                self._wl(f"  ❌ 转化率 {c:.1f}% — 偏低", "bad")
                self._wl("     需重点关注，建议检查商品质量和定价策略", "explain")

            if pv > 20:
                self._wl(f"  ✅ 浏览深度 {pv:.1f} 次 — 用户活跃", "good")
            elif pv > 10:
                self._wl(f"  ⚠️ 浏览深度 {pv:.1f} 次 — 一般", "warn")
            else:
                self._wl(f"  ❌ 浏览深度 {pv:.1f} 次 — 较浅", "bad")

            if cr > 50:
                self._wl(f"  ✅ 券使用率 {cr:.1f}% — 券策略有效", "good")
            elif cr > 30:
                self._wl(f"  ⚠️ 券使用率 {cr:.1f}% — 可优化", "warn")
            else:
                self._wl(f"  ❌ 券使用率 {cr:.1f}% — 需调整", "bad")

        self._render("dashboard", go)

    # ── 用户画像 ──────────────────────────────────────────
    def _show_user(self):
        def go(s):
            self._wl("👤 用户画像", "title"); self._hr(); self._wl()
            self._wl("基础数据", "sec")
            self._metric("总用户", f"{s.get('total_users',0):,} 人")
            self._metric("平均年龄", f"{s.get('age_mean',0):.1f} 岁")
            self._metric("年龄中位数", f"{s.get('age_median',0):.1f} 岁")
            self._metric("平均购买频次", f"{s.get('avg_purchase_freq',0):.1f} 次")
            self._metric("平均消费金额", f"¥{s.get('avg_total_spend',0):,.2f}")
            self._metric("平均粉丝数", f"{s.get('avg_fans',0):.1f}")
            self._metric("平均关注数", f"{s.get('avg_follow',0):.1f}")
            self._wl()

            g = s.get('gender_ratio', {})
            m, f = g.get('男', 0), g.get('女', 0)
            total = m + f
            if total > 0:
                self._wl("性别分布", "sec")
                self._wl(f"  男性: {m} 人 ({m/total*100:.1f}%)")
                self._wl(f"  女性: {f} 人 ({f/total*100:.1f}%)")
                if m > f * 1.5:
                    self._wl("  📌 男性用户为主，可偏向男性偏好商品", "warn")
                elif f > m * 1.5:
                    self._wl("  📌 女性用户为主，可加大美妆服饰投入", "warn")
                else:
                    self._wl("  📌 男女比例均衡，策略可兼顾两性", "good")
            self._wl()

            rfm = s.get('rfm_distribution', {})
            if rfm:
                self._wl("RFM 用户分层", "sec")
                self._wl("  (基于最近购买、购买频次、消费金额)", "dim")
                for lv, cnt in rfm.items():
                    pct = cnt / s.get('total_users', 1) * 100
                    self._wl(f"  {lv}: {cnt} 人 ({pct:.1f}%)")
                self._wl()
                self._wl("分层运营建议", "tip")
                self._wl("  高价值用户: 专属客服、VIP权益、新品优先体验", "explain")
                self._wl("  一般用户: 定期推送、适度折扣、提升复购", "explain")
                self._wl("  流失风险: 召回活动、大额优惠券、限时特价", "explain")
            self._wl()

            # 用户价值分层
            vs = s.get('value_segmentation', {})
            if vs:
                self._wl("用户价值分层", "sec")
                self._wl("  (基于购买频次40% + 消费金额30% + 活跃度30%)", "dim")
                for lv, cnt in vs.items():
                    pct = cnt / s.get('total_users', 1) * 100
                    self._wl(f"  {lv}: {cnt} 人 ({pct:.1f}%)")
                self._wl()
                self._wl("运营策略", "tip")
                self._wl("  高价值用户: 深度维护，提供专属服务", "explain")
                self._wl("  潜力用户: 引导消费升级，提升复购", "explain")
                self._wl("  低活跃用户: 召回活动，激活沉睡用户", "explain")
        self._render("user", go)

    # ── 商品分析 ──────────────────────────────────────────
    def _show_product(self):
        def go(s):
            self._wl("📦 商品表现分析", "title"); self._hr(); self._wl()
            self._wl("基础数据", "sec")
            self._metric("平均价格", f"¥{s.get('avg_price',0):.2f}")
            self._metric("平均折扣率", f"{s.get('avg_discount',0)*100:.1f}%")
            self._wl()

            v = s.get('video_impact', {})
            if v:
                self._wl("视频对购买率的影响", "sec")
                yes, no = v.get('有视频购买率', 0), v.get('无视频购买率', 0)
                self._wl(f"  有视频: {yes:.1f}%    无视频: {no:.1f}%")
                diff = yes - no
                if diff > 2:
                    self._wl(f"  📌 视频显著提升购买率 (+{diff:.1f}%)", "good")
                    self._wl("     建议: 增加商品视频内容", "explain")
                elif diff > 0:
                    self._wl(f"  📌 视频有轻微正面影响 (+{diff:.1f}%)", "warn")
                else:
                    self._wl(f"  📌 视频影响不明显", "bad")
            self._wl()

            cat = s.get('category_stats', {})
            if cat and 'purchase_rate' in cat:
                self._wl("各类目购买率", "sec")
                sorted_cats = sorted(cat['purchase_rate'].items(), key=lambda x: x[1], reverse=True)
                for c, r in sorted_cats:
                    self._wl(f"  {c}:  {r*100:.1f}%")
                best = sorted_cats[0][0] if sorted_cats else ""
                self._wl()
                self._wl(f"  📌 {best} 类目转化率最高，建议重点投入", "tip")

            self._wl()

            # 商品热度指数
            hot = s.get('top10_hot_items', [])
            if hot:
                self._wl("商品热度指数", "sec")
                self._wl("  (浏览量×0.2 + 收藏×0.2 + 点赞×0.2 + 评论×0.2 + 购买×0.2)", "dim")
                self._wl(f"  平均热度: {s.get('avg_hotness', 0):.1f}")
                self._wl()
                self._wl("  Top10 热门商品:", "sec")
                for i, item in enumerate(hot[:5], 1):
                    self._wl(f"    {i}. 商品{item['item_id'][-4:]}  热度: {item['hotness']:.1f}", "explain")
            self._wl()

            # 基于热度指数的建议
            if s.get('top10_hot_items'):
                self._wl("数据建议", "tip")
                self._wl("  → 高热度商品应增加曝光和库存", "explain")
        self._render("product", go)

    # ── 优惠券 ──────────────────────────────────────────
    def _show_coupon(self):
        def go(s):
            self._wl("🎫 优惠券策略分析", "title"); self._hr(); self._wl()
            self._wl("核心指标", "sec")
            self._metric("领券人数", f"{s.get('total_received',0):,} 人")
            self._metric("用券人数", f"{s.get('total_used',0):,} 人")
            self._metric("券使用率", f"{s.get('usage_rate',0):.1f}%")
            self._metric("未用券人数", f"{s.get('unused_user_count',0):,} 人")
            self._metric("最优折扣区间", s.get('best_discount_range', 'N/A'))
            self._metric("最优区间转化率", f"{s.get('best_discount_rate',0):.1f}%")
            self._wl()

            u = s.get('usage_rate', 0)
            self._wl("使用率评估", "sec")
            if u > 60:
                self._wl(f"  ✅ {u:.1f}% — 优秀，券策略有效", "good")
            elif u > 40:
                self._wl(f"  ⚠️ {u:.1f}% — 中等，可优化发放人群", "warn")
            else:
                self._wl(f"  ❌ {u:.1f}% — 偏低，需调整策略", "bad")
            self._wl()

            unused = s.get('unused_user_profile', {})
            if unused:
                self._wl("领券未用用户画像", "sec")
                self._wl(f"  平均年龄: {unused.get('avg_age',0):.1f} 岁")
                self._wl(f"  平均等级: {unused.get('avg_level',0):.1f}")
                self._wl(f"  平均购买频次: {unused.get('avg_freq',0):.1f} 次")
                self._wl()
                self._wl("  📌 这些用户有购买意愿但未转化", "warn")
                self._wl("     建议: 到期前推送提醒或追加折扣", "explain")
            self._wl()

            sug = s.get('retention_suggestions', [])
            if sug:
                self._wl("再触达策略", "tip")
                for x in sug:
                    self._wl(f"  • {x}")
        self._render("coupon", go)

    # ── 消费倾向分析 ──────────────────────────────────────────
    def _show_prediction(self):
        def go(s):
            self._wl("🔮 消费倾向分析", "title"); self._hr()
            self._wl("  基于统计模型的用户购买行为预测", "dim")
            self._wl()
            self._wl("模型信息", "sec")
            self._metric("最优模型", s.get('best_model', 'N/A'))
            self._metric("训练集", f"{s.get('train_size',0):,} 条")
            self._metric("测试集", f"{s.get('test_size',0):,} 条")
            self._wl()

            r = s.get('model_results', {})
            if r:
                self._wl("模型对比（通俗解读）", "sec")
                self._wl("  " + "─" * 40, "dim")
                for n, m in r.items():
                    f1 = m.get('f1', 0)
                    auc = m.get('auc', 0)
                    grade = "优秀 ✅" if f1 > 0.7 else ("良好 ⚠️" if f1 > 0.6 else "一般 ❌")
                    self._wl(f"  {n}:")
                    self._wl(f"    准确率 {m.get('accuracy',0):.1%} | F1 {f1:.3f} | AUC {auc:.3f} → {grade}", "explain")
                self._wl()
                self._wl("  指标说明:", "dim")
                self._wl("  • 准确率: 预测对的比例", "dim")
                self._wl("  • F1: 综合评价（越高越好，>0.7优秀）", "dim")
                self._wl("  • AUC: 区分能力（>0.75良好）", "dim")
            self._wl()

            feat = s.get('top_features', {})
            if feat:
                self._wl("影响购买的关键因素", "sec")
                self._wl("  (按重要性排序，数值越大影响越强)", "dim")
                self._wl()
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
                for i, (k, v) in enumerate(feat.items(), 1):
                    cn = feat_names.get(k, k)
                    # 重要性等级
                    if v > 0.5:
                        level = "🔴 极高"
                    elif v > 0.1:
                        level = "🟠 高"
                    elif v > 0.01:
                        level = "🟡 中"
                    else:
                        level = "🟢 低"
                    self._wl(f"  {i:2d}. {cn:<12} {v:.4f}  {level}")

                self._wl()
                self._wl("业务解读", "tip")
                # 只解读最重要的3个特征
                top_feats = list(feat.keys())[:3]
                feat_tips = {
                    "add2cart": "已加购用户是最有可能下单的，应重点跟进",
                    "coupon_used": "优惠券能有效促进转化",
                    "is_follow_author": "关注作者的用户购买意愿更强",
                    "interaction_rate": "高互动用户购买意愿更强",
                    "user_level": "用户等级越高，购买概率越大",
                    "purchase_freq": "老客户复购率更高",
                    "pv_count": "浏览越多，购买概率越大",
                    "price": "价格对购买决策有显著影响",
                    "discount_rate": "折扣力度影响购买意愿",
                }
                for k in top_feats:
                    cn = feat_names.get(k, k)
                    self._wl(f"  • {cn}: {feat_tips.get(k, '对购买决策有显著影响')}", "explain")
        self._render("prediction", go)

    # ── 社交影响力 ──────────────────────────────────────────
    def _show_social(self):
        def go(s):
            self._wl("📢 社交影响力分析", "title"); self._hr(); self._wl()
            self._wl("概况", "sec")
            self._metric("高影响力用户", f"{s.get('total_influencers',0):,} 人")
            self._metric("平均影响力评分", f"{s.get('avg_influence_score',0):.1f} / 100")
            self._wl("  评分公式: 粉丝20% + 互动率30% + 分享20% + 转化率30%", "dim")
            self._wl()

            top = s.get('top10_users', [])
            if top:
                self._wl("Top5 影响力用户", "sec")
                self._wl("  " + "─" * 40, "dim")
                for i, u in enumerate(top[:5], 1):
                    uid = u.get('user_id', '')
                    score = u.get('influence_score', 0)
                    fans = u.get('fans_num', 0)
                    rate = u.get('avg_interaction_rate', 0)
                    self._wl(f"  {i}. 用户 {uid[-6:]}")
                    self._wl(f"     影响力: {score:.1f}  粉丝: {fans}  互动率: {rate:.1f}", "explain")
                self._wl()

            cats = s.get('high_influence_categories', {})
            if cats:
                self._wl("高影响力用户偏好类目", "sec")
                for cat, cnt in cats.items():
                    self._wl(f"  • {cat}: {cnt} 次互动")
                self._wl()

            sug = s.get('promotion_suggestions', [])
            if sug:
                self._wl("KOL 合作建议", "tip")
                for x in sug[:3]:
                    uid = x.get('user_id', '')
                    self._wl(f"  • 用户 {uid[-6:]}: {x.get('suggestion','')}")

            self._wl()

            # 商品社交影响力
            sid = s.get('social_influence_distribution', {})
            if sid:
                self._wl("商品社交影响力分层", "sec")
                self._wl("  (点赞×0.3 + 评论×0.3 + 分享×0.4)", "dim")
                for lv, cnt in sid.items():
                    self._wl(f"  {lv}: {cnt} 个商品")
                self._wl()
                self._wl("运营建议", "tip")
                self._wl("  高影响商品: 加大推广投入，扩大传播", "explain")
                self._wl("  普通商品: 优化内容，提升互动", "explain")
                self._wl("  低影响商品: 分析原因，调整策略", "explain")
            self._wl()

            # 基于数据的建议
            if sug:
                self._wl("数据建议", "tip")
                self._wl("  → 优先与 Top KOL 合作推广新品", "explain")
                if cats:
                    self._wl(f"  → 针对 {list(cats.keys())[0]} 类目做精准投放", "explain")
        self._render("social", go)

    def _export(self):
        if not self.all_stats:
            messagebox.showwarning("提示", "请先运行分析")
            return
        try:
            path = self.exporter.export_all(self.all_stats, self.all_charts)
            messagebox.showinfo("成功", f"已导出:\n{path}")
            self._open_output()
        except Exception as e:
            messagebox.showerror("失败", str(e))

    def _open_output(self):
        if os.path.exists(OUTPUT_DIR):
            if sys.platform == "win32":
                os.startfile(OUTPUT_DIR)
            elif sys.platform == "darwin":
                subprocess.run(["open", OUTPUT_DIR])
            else:
                subprocess.run(["xdg-open", OUTPUT_DIR])


if __name__ == "__main__":
    App().run()
