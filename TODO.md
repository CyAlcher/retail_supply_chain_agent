# TODO — 下一步开发优化点

> 版本：v0.1.0 演示版  
> 更新日期：2026-05-10  
> 原则：按 P0/P1/P2 优先级排列，P0 影响演示可信度，P1 影响面试深度，P2 影响生产可用性

---

## P0 — 影响演示可信度（下次开发必做）

### T-01  接入真实 Claude API（意图解析）
- **现状**：`entrypoint.py` 用关键词规则匹配，无法处理复杂自然语言
- **目标**：用 `anthropic.messages.create` + `tool_use` 做结构化槽位抽取
- **文件**：`src/retail_agent/layer1_perception/entry/entrypoint.py`
- **验证**：输入"帮我看看下周三华南区冰饮促销备货够不够"能正确解析门店/SKU/日期

### T-02  WhatIf 促销弹性为 0 的问题
- **现状**：DEV-PROMO-002 两个方案销量相同（弹性 0.0%），因为合成数据样本量不足
- **目标**：增加合成数据量（500 条以上），或在 WhatIfAgent 里直接用弹性系数公式
- **文件**：`data/promo_synthetic/gen_data.py`、`src/retail_agent/layer2_orchestration/experts/what_if.py`
- **验证**：DEV-PROMO-002 的方案 B 销量必须 > 方案 A 销量

### T-03  CriticAgent 接入 LLM-as-Judge
- **现状**：规则阈值判断（区间宽度 + 覆盖天数），无法识别语义层面的质量问题
- **目标**：用 Claude 做六维评分（准确性/完备性/合规性/可执行性/过程合理性/业务价值）
- **文件**：`src/retail_agent/layer2_orchestration/experts/critic.py`
- **验证**：对故意构造的低质量输出，Critic 能给出 < 0.6 的评分并触发 retry

---

## P1 — 影响面试深度（1-2 周内）

### T-04  ExplainAgent 接入 Claude 生成自然语言
- **现状**：模板字符串拼接，叙述千篇一律
- **目标**：Claude 根据归因因子生成差异化的业务语言叙述
- **文件**：`src/retail_agent/layer2_orchestration/experts/explain.py`

### T-05  补全开发集 DEV-PROMO-003/004/005
- **现状**：只跑通了 001 和 002，003（偏差归因）、004（多门店）、005（天气异常）未实现
- **目标**：实现归因分解工具、多门店并行预测、气象预警降档逻辑
- **文件**：新增 `src/retail_agent/layer3_compute/attribution/tool.py`

### T-06  LangGraph Checkpointer 替换为 SQLiteSaver
- **现状**：MemorySaver，进程退出即丢失，无法演示决策回放
- **目标**：换成 `SqliteSaver`（无需 Postgres，本地文件即可），支持 `--replay task_id`
- **文件**：`src/retail_agent/layer2_orchestration/planner/planner.py`
- **验证**：`demo/run_demo.py --replay DEV-PROMO-001` 能重放历史决策

### T-07  ForecastAgent 模型路由接入 Claude tool_use
- **现状**：直接调 ML tool，跳过 LLM 路由决策
- **目标**：Claude 根据场景特征（促销/季节/长尾）动态选择统计族/ML族/深度族
- **文件**：`src/retail_agent/layer2_orchestration/experts/forecast.py`

### T-08  添加 TimesFM 零样本预测作为对比基线
- **现状**：只有 LightGBM，缺少基础模型对比
- **目标**：接入 TimesFM，在 DEV-PROMO-001 上对比 LightGBM vs TimesFM 的 MAPE
- **文件**：`src/retail_agent/layer3_compute/forecast_engine/deep/tool.py`（新增）

---

## P2 — 影响生产可用性（Phase 2）

### T-09  HITLGate 实现真实审核界面
- **现状**：自动通过，无法演示人工介入
- **目标**：Gradio 简单界面，展示预测结果 + 风险提示，支持"批准/拒绝/修改"
- **文件**：`src/retail_agent/governance/hitl/gate.py`、新增 `demo/hitl_ui.py`

### T-10  Arize Phoenix 可观测性接入
- **现状**：终端打印，无法追踪 LLM 调用链路
- **目标**：接入 Phoenix 自托管，记录每次 Agent 调用的 span/trace/token 消耗
- **文件**：`src/retail_agent/governance/audit/logger.py`

### T-11  VersionRegistry + Rollbacker 最小实现
- **现状**：无版本管理，模型/Prompt 变更无法回滚
- **目标**：JSON 文件记录模型版本 + Prompt 版本，`--rollback v0.1.0` 一键恢复
- **文件**：新增 `src/retail_agent/versioning/`

### T-12  多门店并行预测（DEV-PROMO-004）
- **现状**：单门店串行
- **目标**：LangGraph Send API 实现 10 家门店并行预测，分层聚合一致性校验
- **文件**：`src/retail_agent/layer2_orchestration/planner/planner.py`

### T-13  蒙特卡洛仿真安全库存（高价值 SKU）
- **现状**：只有 Z-Score，假设需求正态分布
- **目标**：对 A 类 SKU 用蒙特卡洛仿真，输出更准确的安全库存区间
- **文件**：`src/retail_agent/layer3_compute/safety_stock_engine/monte_carlo/`（新增）

### T-14  接入真实 KA 客户历史数据
- **现状**：合成数据，统计特征真实但非真实业务数据
- **目标**：Feature Store 接口，支持按 sku_id + store_id + date_range 查询
- **文件**：`src/retail_agent/layer1_perception/context/builder.py`

---

## 已知问题（需在演示前说明）

| 问题 | 影响 | 说明口径 |
|---|---|---|
| WhatIf 促销弹性为 0 | DEV-PROMO-002 两方案销量相同 | 合成数据样本量不足，真实 KA 数据下弹性系数会显著分离 |
| Critic 可执行性评分 0.6 | 质量总分被拉低 | 演示版 ExplainAgent 用模板，接入 Claude 后可执行性会提升到 0.9+ |
| 补货提前期 hardcode 30 天 | 不同 SKU 提前期不同 | TODO T-14 接入商品主数据后动态读取 |
