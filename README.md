<div align="center">

# retail-supply-chain-agent

**AI-Native 零售供应链安全库存 Agent — 促销场景演示版（v0.1.0）**

> 把"促销期备多少货"从人工经验判断升级为可解释、可审计、可回滚的 AI-Native 闭环。

[English](#english-tldr) ·
[快速开始](#快速开始) ·
[架构设计](#架构设计) ·
[项目结构](#项目结构) ·
[Mock 边界](#mock-vs-真实边界)

</div>

---

基于 LangGraph 的 5-Expert + Critic 多智能体架构，以**促销场景**为切入点，端到端覆盖销量预测、安全库存计算、What-if 方案推演、归因解释、质量评估完整决策链。

---

## 背景

零售供应链安全库存的核心挑战：

- **促销期需求剧烈波动**：折扣率、天气、节假日、竞品同时影响销量，传统规则无法动态响应
- **备货决策不可解释**：业务方不信任黑箱预测，需要"为什么备这么多"的可读归因
- **多因子联动**：促销力度 × 天气 × 门店特征 × 历史 uplift，需要多 Expert 协同而非单点模型
- **质量评估缺失**：预测结果没有置信度分级，高风险场景无法自动触发人工审核

本项目用 AI-Native 架构解决上述问题，Critic Agent 承担质量评估与风险识别，HITL Gate 在低置信场景强制人工介入。

---

## 快速开始

**环境要求**：Python 3.10+，conda 环境 `agent`

```bash
# 1. 生成合成促销历史数据（首次运行，约 3 秒）
conda run -n agent python data/promo_synthetic/gen_data.py

# 2. 单品促销预测 + 安全库存 + 补货建议（DEV-PROMO-001）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-001

# 3. 折扣力度对比推演（DEV-PROMO-002）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-002

# 4. 带硬断言验证（CI 门禁用）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-001 --validate

# 5. 查看 Mock / Hardcode / TODO 清单
conda run -n agent python demo/run_demo.py --show-todos
```

**典型输出**：

```
[Planner]     意图识别 → 促销预测+补货建议  场景路由 → D4_promo
[Critic①]    计划合理性校验 → ✓ 通过
[Forecast]    ML预测工具 → P25=2262  P50=2294  P75=2481  模型=lgb_quantile_uplift
[SafetyStock] Z-Score → 安全库存=1040件  覆盖天数=30.0  服务水平=95%
[WhatIf]      方案A(25%折扣) 毛利¥1548 vs 方案B(35%折扣) 毛利¥516 → 推荐方案A
[Explain]     促销uplift+20% / 天气+10.5% / 周末效应+18%
[Critic②]    质量评分=0.90  风险=无  决策=accept
[Action]      下单 3334件  置信度=自动执行
```

---

## 架构设计

对应设计文档 `AI-Native-落地规划.md`（5 Expert + Critic + 版本回滚层）。

```
自然语言输入
    │
    ▼
┌─────────────────────────────────────────┐
│ Layer1  感知层                           │
│  TaskEntrypoint  — 槽位解析              │
│  ContextBuilder  — 历史数据 + 未来因子   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Layer2  编排层（LangGraph StateGraph）   │
│                                         │
│  PlannerAgent   — 意图路由 + 计划生成    │
│  Critic①       — 计划合理性前置校验     │
│                                         │
│  ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │ Forecast │ │SafetyStk │ │ WhatIf  │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│  ┌──────────┐                           │
│  │ Explain  │                           │
│  └──────────┘                           │
│                                         │
│  Critic②  — 质量评分+风险识别+反思决策  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Layer3  计算层                           │
│  LightGBM 分位数回归（P25/P50/P75）      │
│  Z-Score 安全库存引擎（SS = z·σ·√L）    │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Layer4  决策层                           │
│  ActionBuilder — 补货建议 + 置信度分级   │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│ Governance                              │
│  HITLGate   — 低置信/高风险强制人工审核  │
│  AuditLogger — 全链路结构化审计日志      │
└─────────────────────────────────────────┘
```

**LangGraph 图路由逻辑**：

```
START → plan → critic_check_plan
      → forecast → safety_stock → [what_if] → critic_score
        ├─ accept      → explain → action_builder → hitl → audit → END
        ├─ retry(≤2)   → 回 forecast 重跑
        └─ escalate    → hitl(人工) → audit → END
```

---

## 安全库存计算说明

核心公式（Z-Score 服务水平法）：

```
SS = z(SL) × σ_d × √L

SS  = 安全库存量（件）
z   = 正态分布分位数（SL=95% → z=1.645）
σ_d = 日需求标准差（从历史数据计算）
L   = 补货提前期（天，默认 30 天月度补货周期）
```

覆盖天数语义：安全库存覆盖补货提前期内的需求波动风险，`coverage_days = L`。

促销期备货量 = 预测中位数（P50）+ 安全库存（SS）。

---

## Mock vs 真实边界

演示版用规则/合成数据替代 LLM 调用，保证离线可运行、不依赖网络。

| 模块 | 演示实现 | 真实化路径 |
|---|---|---|
| 意图解析 | 关键词规则匹配 | Claude API `tool_use` slot-filling |
| ForecastAgent 模型路由 | 直接调 ML tool | Claude `tool_use` 动态选模型族 |
| CriticAgent 评分 | 规则阈值（WAPE + 区间宽度） | Claude LLM-as-Judge + 结构化 Rubric |
| ExplainAgent 叙述 | 模板字符串拼接 | Claude 生成自然语言归因 |
| HITLGate | 自动通过 | 真实审核界面（Web/IM） |
| LangGraph Checkpointer | MemorySaver（内存） | PostgresSaver（持久化 + 决策回放） |
| 训练数据 | 合成数据（统计特征真实） | 真实 KA 客户历史销量 Feature Store |
| 可观测性 | 终端打印 | Arize Phoenix + OpenTelemetry |

**真实实现的部分（非 mock）**：
- LightGBM 分位数回归：在合成数据上真实训练，输出 P25/P50/P75
- Z-Score 安全库存：`SS = z × σ × √L`，纯数学，无近似
- LangGraph StateGraph：真实图结构，含条件路由、retry 上限、escalate 分支
- Critic 规则判断：真实逻辑（区间宽度阈值 + 覆盖天数检查），非随机
- 合成数据：促销 uplift 系数、天气系数、周末系数统计特征真实

---

## 开发集验证

| Case | 描述 | 硬断言 | 状态 |
|---|---|---|---|
| DEV-PROMO-001 | 单品促销预测 + 安全库存 + 补货建议 | P50∈[1800,2600]、覆盖天数≥10、置信度≠拒绝 | ✅ 全部通过 |
| DEV-PROMO-002 | 折扣力度对比推演（20% vs 30%） | WhatIf 有 2 个方案、推荐方案非空 | ✅ 全部通过 |

完整开发集（5 个促销场景）见 `evals/dev/dev.yaml`，测试集（MVP 签字门槛）见 `evals/dev/test.yaml`。

---

## 项目结构

```
retail_supply_chain_agent/
├── src/retail_agent/
│   ├── schemas/core.py                          # Pydantic 数据契约（跨层共享）
│   ├── layer1_perception/
│   │   ├── entry/entrypoint.py                  # 自然语言槽位解析
│   │   └── context/builder.py                   # 历史数据 + 未来因子聚合
│   ├── layer2_orchestration/
│   │   ├── planner/planner.py                   # LangGraph StateGraph 主图
│   │   └── experts/
│   │       ├── base.py                          # Expert 抽象基类
│   │       ├── forecast.py                      # ForecastAgent
│   │       ├── safety_stock.py                  # SafetyStockAgent
│   │       ├── what_if.py                       # WhatIfAgent
│   │       ├── explain.py                       # ExplainAgent
│   │       └── critic.py                        # CriticAgent（6 项职责）
│   ├── layer3_compute/
│   │   ├── forecast_engine/ml/tool.py           # LightGBM 分位数回归
│   │   └── safety_stock_engine/z_score/engine.py # Z-Score 安全库存
│   ├── layer4_decision/action_builder/builder.py # 补货建议 + 置信度分级
│   └── governance/
│       ├── hitl/gate.py                         # HITL 审核闸门
│       └── audit/logger.py                      # 全链路审计日志
├── data/promo_synthetic/gen_data.py             # 合成促销历史数据生成器
├── evals/dev/
│   ├── dev.yaml                                 # 开发集（5 个促销场景）
│   └── test.yaml                                # 测试集（MVP 签字门槛）
├── demo/run_demo.py                             # 命令行演示入口
├── pyproject.toml
└── TODO.md                                      # 下一步开发优化点
```

---

## 依赖

```
langgraph>=0.2  langchain-core>=0.3  pydantic>=2.0
lightgbm>=4.3   scipy>=1.12          pandas>=2.0
fastapi>=0.110  uvicorn>=0.29        anthropic>=0.40
```

安装：`pip install -e .`（需要 conda 环境中已有 lightgbm，建议 `conda install -c conda-forge lightgbm`）

---

## Contributing

Issues and PRs welcome. Please do not commit any real customer data or API keys.

---

## License

MIT

---

## English TL;DR

**retail-supply-chain-agent** is an AI-Native safety-stock agent for retail supply chains, focused on promotional scenarios.

Built on a **LangGraph 5-Expert + Critic** multi-agent architecture, it covers the full decision chain end-to-end: demand forecasting → safety stock calculation → what-if scenario analysis → attribution explanation → quality assessment.

**Core value**: Turns "how much inventory to stock for a promotion" — the highest-frequency, highest-value decision in retail supply chains — from manual rule-of-thumb into an explainable, auditable, rollback-capable AI-Native loop.

**Key design points**:
- LightGBM quantile regression (P25/P50/P75) trained on synthetic promo data with realistic uplift coefficients
- Z-Score safety stock engine: `SS = z × σ × √L`, no approximations
- Critic Agent handles quality scoring, risk identification, and retry/escalate routing
- HITL Gate forces human review on low-confidence or high-risk decisions
- Full audit log across all layers

**Stack**: LangGraph · LightGBM · Pydantic v2 · Claude API (roadmap) · Arize Phoenix (roadmap)

> Related: [CyAlcher/ai_native_a_stock_agent](https://github.com/CyAlcher/ai_native_a_stock_agent) · [CyAlcher/CompoundMe](https://github.com/CyAlcher/CompoundMe)
