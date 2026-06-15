# E-Commerce Analytics

> Python Course Project — Social E-commerce Data Analysis & Decision Support

## 这是什么

一个基于 Python 的社交电商数据分析工具，帮商家搞清楚三件事：

1. **我的客户是谁** — 用户画像、消费分层
2. **什么商品好卖** — 类目热度、价格区间、折扣效果
3. **券发给谁最有效** — A/B 测试、再触达策略

底层跑了一个简单的机器学习模型，能预测用户购买概率，输出高/中/低意愿等级。

## 数据来源

数据来自天池数据集：

https://tianchi.aliyun.com/dataset/215680

包含 10 万条用户行为记录，32 个字段，覆盖用户画像、商品属性、互动行为、购买标签。

## 功能模块

| 模块 | 做什么 |
|------|--------|
| 数据概览 | 核心指标总览：用户数、转化率、GMV |
| 用户画像 | 年龄/性别分布、RFM 分层、价值分层 |
| 商品分析 | 类目热度、价格区间、折扣曲线、热度指数 |
| 优惠券策略 | 使用率分析、A/B 测试矩阵、再触达建议 |
| 消费倾向 | 三模型对比（逻辑回归/随机森林/XGBoost）|
| 社交影响力 | KOL 识别、影响力评分、类目偏好 |

## 怎么跑

```bash
# 装依赖
pip install -r requirements.txt

# 启动
python main.py
```

启动后会自动读取 `data/` 文件夹里的 CSV，点左边的模块按钮切换分析页面。

## 项目结构

```
├── main.py              # 主程序（GUI）
├── config.py            # 配置
├── requirements.txt     # 依赖
├── data/                # 放 CSV 数据
└── modules/
    ├── data_loader.py   # 数据导入与清洗
    ├── dashboard.py     # 数据概览
    ├── user_analysis.py # 用户画像
    ├── product_analysis.py # 商品分析
    ├── coupon_analysis.py  # 优惠券策略
    ├── prediction.py    # 消费倾向预测
    ├── social_analysis.py  # 社交影响力
    ├── exporter.py      # Excel 报告导出
    └── utils.py         # 公共工具
```

## 用了什么技术

- **Pandas** — 数据处理
- **Matplotlib + Seaborn** — 可视化
- **Scikit-learn + XGBoost** — 机器学习
- **Tkinter** — 桌面 GUI
- **SQLite** — 数据存储

## 一些设计说明

**关于权重：**

商品热度指数和影响力评分的权重不是拍脑袋定的，而是根据业务逻辑调整的：

- 热度指数：购买 0.35 > 评论 0.2 > 收藏 0.2 > 点赞 0.15 > 浏览 0.1
  - 购买含金量最高，浏览门槛最低
- 影响力评分：分享 0.3 = 转化率 0.3 > 互动率 0.25 > 粉丝 0.15
  - 分享是社交传播的核心，粉丝数容易刷

**关于 RFM 分层：**

- R (最近活跃)：3 天内 = 高，14 天内 = 中，其他 = 低
- F (购买频次)：15+ = 高，5+ = 中，其他 = 低
- M (消费金额)：5000+ = 高，1500+ = 中，其他 = 低

## 已知问题

- GUI 界面比较朴素，没花时间美化
- 数据是静态 CSV，没有接实时数据源
- 模型训练需要手动触发，没有自动化

## License

MIT
