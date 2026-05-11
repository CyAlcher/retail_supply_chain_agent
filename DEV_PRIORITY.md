# 开发优先级清单

> 基于 EVAL_REPORT.md 评测结论  
> 更新日期：2026-05-12  
> 原则：ROI 优先，每项有明确验收标准

---

## P0 — 影响演示可信度（本周必做）

### P0-1  API 超时熔断（预计 2 小时）
- **问题**：entrypoint / critic / explain / forecast 的 LLM 调用无超时，API 慢时整个链路挂起
- **方案**：所有 `client.messages.create` 加 `timeout=30.0` 参数
- **文件**：`entrypoint.py` / `critic.py` / `explain.py` / `forecast.py`
- **验收**：模拟 API 超时时，fallback 规则正常接管，demo 不挂起

### P0-2  Critic 风险误报修复（预计 1 小时）
- **问题**：Critic 风险列表包含"action字段为null"——action 在 Critic 之后生成，属时序误报
- **方案**：在 `_llm_score` 的 context 里注明 action 字段在 Critic 之后生成，或在 prompt 里过滤该误报
- **文件**：`critic.py`
- **验收**：DEV-PROMO-001 的 Critic 风险列表不再出现"action字段为null"

### P0-3  录制 HITL 演示（预计 1 小时）
- **问题**：HITL_AUTO_APPROVE=1 默认跳过，演示时无法展示人工介入价值
- **方案**：关闭 HITL_AUTO_APPROVE，跑一次 DEV-PROMO-005（暴雨预警，置信度=需要复核），截图 Gradio UI
- **文件**：`gate.py`（确认 UI 正常弹出）
- **验收**：Gradio 审核界面能正常弹出，批准/拒绝按钮可用

---

## P1 — 影响评测可信度（本周内）

### P1-1  Critic 校准集（预计 1 天）
- **问题**：LLM-as-Judge 评分无人工标注对照，可信度存疑
- **方案**：构建 15 条标注样本（5好/5中/5差），写入 `evals/critic_calibration.yaml`，跑一次一致性检验
- **文件**：新增 `evals/critic_calibration.yaml` + `evals/run_calibration.py`
- **验收**：LLM 评分与人工标注的 Spearman 相关系数 ≥ 0.7

### P1-2  Phoenix 可观测性真实演示（预计 2 小时）
- **问题**：Phoenix trace 代码完整但未真实运行过
- **方案**：`PHOENIX_ENABLED=1` 启动 Phoenix server，跑 DEV-PROMO-001，截图 trace 界面
- **文件**：`logger.py`（验证 span 写入正确）
- **验收**：Phoenix UI 能看到 task_id / forecast.p50 / critic.score 等 span 属性

### P1-3  多门店真实区分（DEV-PROMO-004）（预计 半天）
- **问题**：DEV-PROMO-004 三家门店用同一数据，预测结果完全相同
- **方案**：在合成数据生成器里为越秀店（基线×0.85）、海珠店（基线×0.75）生成差异化数据
- **文件**：`gen_data.py` / `context/builder.py`
- **验收**：DEV-PROMO-004 三家门店 P50 有明显差异（>10%）

---

## P2 — 影响生产可用性（下周）

### P2-1  真实数据接入（预计 2 天）
- **问题**：合成数据无法计算真实 MAPE，预测价值无法量化
- **方案**：接入至少 1 个 SKU 的 90 天真实销量（CSV 格式），计算 LGB vs 统计基线的真实 MAPE
- **文件**：`gen_data.py`（替换或并行）/ `tool.py`（增加 MAPE 计算）
- **验收**：能输出"LGB MAPE=X%，统计基线 MAPE=Y%，LGB 优于基线 Z%"

### P2-2  LangGraph Send API 多门店并行（预计 1 天）
- **问题**：多门店串行，无法体现 LangGraph 并行能力
- **方案**：用 LangGraph Send API 实现 3 家门店并行预测，汇总结果
- **文件**：`planner.py`
- **验收**：DEV-PROMO-004 输出 3 家门店各自的 P50 + 汇总备货量

### P2-3  学习闭环最小实现（预计 1 天）
- **问题**：持续进化要素完成度仅 40%，无销量回传和模型迭代
- **方案**：实现 `feedback/` 模块，接收实际销量，计算 MAPE，触发模型重训标记
- **文件**：新增 `learning_loop/feedback/collector.py`
- **验收**：输入实际销量后，能输出"预测偏差 X%，建议重训"

---

## 优先级矩阵

```
高影响
  │
  │  P0-1(熔断)   P1-1(校准集)
  │  P0-2(误报)   P1-3(多门店)
  │  P0-3(HITL)   P1-2(Phoenix)
  │
  │              P2-1(真实数据)
  │              P2-2(并行)
  │              P2-3(学习闭环)
  │
低影响
  └──────────────────────────────
     低复杂度              高复杂度
```

**本次开发顺序**：P0-1 → P0-2 → P1-3 → P1-1 → P2-2
