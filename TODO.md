# TODO — 下一步开发优化点

> 版本：v0.2.0  
> 更新日期：2026-05-12  
> 原则：按 P0/P1/P2 优先级排列，P0 影响演示可信度，P1 影响面试深度，P2 影响生产可用性

---

## P0 — 影响演示可信度 ✅ 已完成

### T-01  接入真实 Claude API（意图解析）✅
- Claude `tool_use` slot-filling，fallback 到规则匹配
- 文件：`src/retail_agent/layer1_perception/entry/entrypoint.py`

### T-02  WhatIf 促销弹性为 0 的问题 ✅
- 弹性系数 1.2x/1%，合成数据扩充至 540 天
- 文件：`data/promo_synthetic/gen_data.py`、`experts/what_if.py`

### T-03  CriticAgent 接入 LLM-as-Judge ✅
- Claude tool_use 六维 Rubric 评分，fallback 到规则
- 文件：`src/retail_agent/layer2_orchestration/experts/critic.py`

---

## P1 — 影响面试深度 ✅ 已完成

### T-04  ExplainAgent 接入 Claude 生成自然语言 ✅
- Claude 生成差异化归因叙述，fallback 到模板
- 文件：`experts/explain.py`

### T-05  补全开发集 DEV-PROMO-003/004/005 ✅
- 003：偏差归因分析；004：多门店并行预测；005：天气异常降档
- 文件：`demo/run_demo.py`

### T-06  LangGraph SqliteSaver 替换 MemorySaver ✅
- 支持 `--replay task_id` 决策回放
- 文件：`planner/planner.py`

### T-07  ForecastAgent 模型路由接入 Claude tool_use ✅
- Claude 根据场景特征动态选择 lgb_quantile_uplift / statistical_baseline
- 文件：`experts/forecast.py`

### T-08  统计基线对比（替代 TimesFM） ✅
- 历史均值 × 促销/天气/周末系数，输出 baseline_p50 + mape_vs_baseline
- 文件：`layer3_compute/forecast_engine/ml/tool.py`

---

## P2 — 影响生产可用性（Phase 2）

### T-09  HITLGate 实现真实审核界面
- **现状**：自动通过，无法演示人工介入
- **目标**：Gradio 简单界面，展示预测结果 + 风险提示，支持"批准/拒绝/修改"
- **文件**：`governance/hitl/gate.py`、新增 `demo/hitl_ui.py`

### T-10  Arize Phoenix 可观测性接入
- **现状**：终端打印，无法追踪 LLM 调用链路
- **目标**：接入 Phoenix 自托管，记录每次 Agent 调用的 span/trace/token 消耗
- **文件**：`governance/audit/logger.py`

### T-11  VersionRegistry + Rollbacker 最小实现
- **现状**：无版本管理，模型/Prompt 变更无法回滚
- **目标**：JSON 文件记录模型版本 + Prompt 版本，`--rollback v0.1.0` 一键恢复
- **文件**：新增 `src/retail_agent/versioning/`

### T-12  多门店并行预测（LangGraph Send API）
- **现状**：单门店串行（DEV-PROMO-004 用同一门店数据模拟）
- **目标**：LangGraph Send API 实现真正并行，分层聚合一致性校验
- **文件**：`planner/planner.py`

### T-13  蒙特卡洛仿真安全库存（高价值 SKU）
- **现状**：只有 Z-Score，假设需求正态分布
- **目标**：对 A 类 SKU 用蒙特卡洛仿真，输出更准确的安全库存区间
- **文件**：新增 `layer3_compute/safety_stock_engine/monte_carlo/`

### T-14  接入真实 KA 客户历史数据
- **现状**：合成数据，统计特征真实但非真实业务数据
- **目标**：Feature Store 接口，支持按 sku_id + store_id + date_range 查询
- **文件**：`layer1_perception/context/builder.py`

---

## 已知问题（演示前说明）

| 问题 | 影响 | 说明口径 |
|---|---|---|
| Critic 风险提示包含"action字段为null" | 显示冗余风险 | action 在 critic 之后生成，属正常时序，非真实风险 |
| 补货提前期 hardcode 30 天 | 不同 SKU 提前期不同 | T-14 接入商品主数据后动态读取 |
| DEV-PROMO-004 多门店用同一数据 | 三家门店预测结果相同 | T-12 接入真实多门店数据后分离 |
