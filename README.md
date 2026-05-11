<div align="center">

# retail-supply-chain-agent

**AI-Native 零售供应链安全库存 Agent — 促销场景演示版（v0.2.0）**

> 把"促销期备多少货"从人工经验判断升级为可解释、可审计、可回滚的 AI-Native 闭环。

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](./LICENSE)
[![stars](https://img.shields.io/github/stars/CyAlcher/retail_supply_chain_agent?style=social)](https://github.com/CyAlcher/retail_supply_chain_agent/stargazers)
[![forks](https://img.shields.io/github/forks/CyAlcher/retail_supply_chain_agent?style=social)](https://github.com/CyAlcher/retail_supply_chain_agent/network/members)
[![issues](https://img.shields.io/github/issues/CyAlcher/retail_supply_chain_agent)](https://github.com/CyAlcher/retail_supply_chain_agent/issues)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Stack](https://img.shields.io/badge/LangGraph%20%7C%20LightGBM%20%7C%20Claude%20API-powered-brightgreen)](#依赖)

[English](#english-version) ·
[快速开始](#快速开始) ·
[架构设计](#架构设计) ·
[真实 vs Mock](#mock-vs-真实边界) ·
[项目结构](#项目结构)

</div>

---

基于 LangGraph 的 **5-Expert + Critic 多智能体架构**，以促销场景为切入点，端到端覆盖销量预测、安全库存计算、What-if 方案推演、归因解释、质量评估完整决策链。

**v0.2.0 新增**：Claude API 已接入意图解析、模型路由、LLM-as-Judge、自然语言归因四个核心模块，SQLite 持久化支持决策回放。

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

**环境要求**：Python 3.10+，conda 环境 `agent`，`ANTHROPIC_API_KEY` 环境变量

```bash
# 1. 克隆并安装
git clone https://github.com/CyAlcher/retail_supply_chain_agent.git
cd retail_supply_chain_agent
pip install -e .
pip install langgraph-checkpoint-sqlite

# 2. 生成合成促销历史数据（首次运行，约 3 秒）
conda run -n agent python data/promo_synthetic/gen_data.py

# 3. 单品促销预测 + 安全库存 + 补货建议（DEV-PROMO-001）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-001

# 4. 折扣力度对比推演（DEV-PROMO-002）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-002

# 5. 带硬断言验证（CI 门禁用）
conda run -n agent python demo/run_demo.py --case DEV-PROMO-001 --validate

# 6. 重放历史决策（SQLite 持久化）
conda run -n agent python demo/run_demo.py --replay DEV-PROMO-001
```

**典型输出**：

```
[Planner]     意图识别 → 促销预测+补货建议  场景路由 → D4_promo
[Critic①]    计划合理性校验 → ✓ 通过
[Forecast]    lgb_quantile_uplift [routed: 促销场景，LightGBM分位数回归]
              P25=2188  P50=2375  P75=2538
[SafetyStock] Z-Score → 安全库存=1106件  覆盖天数=30.0  服务水平=95%
[WhatIf]      方案A(25%折扣) 毛利¥1603 vs 方案B(35%折扣) 毛利¥628 → 推荐方案A
[Explain]     本次预测2375件主要由三个因素驱动：基线贡献51.5%，25%促销uplift+20%，
              32℃高温+10.5%，周末效应+18%。
[Critic②]    质量评分=0.88  决策=accept
[Action]      下单 3482件  置信度=需要复核
```

---

## 架构设计

```
自然语言输入
    │
    ▼
┌─────────────────────────────────────────┐
│ Layer1  感知层                           │
│  TaskEntrypoint  — Claude tool_use 槽位解析 │
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
│  │ (Claude  │ │          │ │         │ │
│  │  路由)   │ │          │ │         │ │
│  └──────────┘ └──────────┘ └─────────┘ │
│  ┌──────────┐                           │
│  │ Explain  │ ← Claude 自然语言归因      │
│  └──────────┘                           │
│                                         │
│  Critic②  — Claude LLM-as-Judge 评分   │
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
│  HITLGate    — 低置信/高风险强制人工审核 │
│  AuditLogger — 全链路结构化审计日志      │
│  SQLiteSaver — 决策持久化 + 回放         │
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

促销期备货量 = 预测中位数（P50）+ 安全库存（SS）。

---

## Mock vs 真实边界

| 模块 | v0.1.0 演示实现 | v0.2.0 当前状态 |
|---|---|---|
| 意图解析 | 关键词规则匹配 | ✅ Claude API `tool_use` slot-filling |
| ForecastAgent 模型路由 | 直接调 ML tool | ✅ Claude `tool_use` 动态选模型族 |
| CriticAgent 评分 | 规则阈值（WAPE + 区间宽度） | ✅ Claude LLM-as-Judge + 结构化 Rubric |
| ExplainAgent 叙述 | 模板字符串拼接 | ✅ Claude 生成自然语言归因 |
| LangGraph Checkpointer | MemorySaver（内存） | ✅ SqliteSaver（持久化 + `--replay`） |
| HITLGate | 自动通过 | 🔲 真实审核界面（Roadmap） |
| 训练数据 | 合成数据 180 天 | ✅ 合成数据扩充至 540 天 |
| 可观测性 | 终端打印 | 🔲 Arize Phoenix（Roadmap） |

**真实实现的部分（非 mock）**：
- LightGBM 分位数回归：在合成数据上真实训练，输出 P25/P50/P75
- Z-Score 安全库存：`SS = z × σ × √L`，纯数学，无近似
- LangGraph StateGraph：真实图结构，含条件路由、retry 上限、escalate 分支
- Claude API：意图解析、模型路由、质量评分、归因叙述均已接入，fallback 到规则

---

## 开发集验证

| Case | 描述 | 硬断言 | 状态 |
|---|---|---|---|
| DEV-PROMO-001 | 单品促销预测 + 安全库存 + 补货建议 | P50∈[1800,2600]、覆盖天数≥10、置信度≠拒绝 | ✅ 全部通过 |
| DEV-PROMO-002 | 折扣力度对比推演（20% vs 30%） | WhatIf 有 2 个方案、推荐方案非空 | ✅ 全部通过 |
| DEV-PROMO-003 | 促销偏差归因分析（实际 vs 预测 -24.5%） | 归因因子非空、置信度≠拒绝 | ✅ 全部通过 |
| DEV-PROMO-004 | 多门店并行预测（天河/越秀/海珠） | 动作类型=下单、置信度≠拒绝 | ✅ 全部通过 |
| DEV-PROMO-005 | 天气异常降档（暴雨橙色预警） | 包含气象风险提示、置信度≠拒绝 | ✅ 全部通过 |

---

## 项目结构

```
retail_supply_chain_agent/
├── src/retail_agent/
│   ├── schemas/core.py                          # Pydantic 数据契约（跨层共享）
│   ├── layer1_perception/
│   │   ├── entry/entrypoint.py                  # Claude tool_use 槽位解析
│   │   └── context/builder.py                   # 历史数据 + 未来因子聚合
│   ├── layer2_orchestration/
│   │   ├── planner/planner.py                   # LangGraph StateGraph + SqliteSaver
│   │   └── experts/
│   │       ├── forecast.py                      # ForecastAgent（Claude 模型路由）
│   │       ├── safety_stock.py                  # SafetyStockAgent
│   │       ├── what_if.py                       # WhatIfAgent（弹性系数修正）
│   │       ├── explain.py                       # ExplainAgent（Claude 自然语言）
│   │       └── critic.py                        # CriticAgent（Claude LLM-as-Judge）
│   ├── layer3_compute/
│   │   ├── forecast_engine/ml/tool.py           # LightGBM 分位数回归
│   │   └── safety_stock_engine/z_score/engine.py
│   ├── layer4_decision/action_builder/builder.py
│   └── governance/
│       ├── hitl/gate.py
│       └── audit/logger.py
├── data/promo_synthetic/gen_data.py             # 合成促销历史数据（540 天）
├── demo/run_demo.py                             # CLI 演示入口（含 --replay）
├── pyproject.toml
└── TODO.md
```

---

## 依赖

```
langgraph>=0.2              langgraph-checkpoint-sqlite>=2.0
langchain-core>=0.3         pydantic>=2.0
lightgbm>=4.3               scipy>=1.12
pandas>=2.0                 anthropic>=0.40
fastapi>=0.110              uvicorn>=0.29
```

安装：`pip install -e .` + `pip install langgraph-checkpoint-sqlite`

---

## Roadmap

- [x] LangGraph 5-Expert + Critic 多智能体架构
- [x] Claude API 接入：意图解析 / 模型路由 / LLM-as-Judge / 自然语言归因
- [x] SQLiteSaver 持久化 + `--replay` 决策回放
- [x] WhatIf 促销弹性修正（弹性系数 1.2x/1%）
- [ ] HITLGate 真实审核界面（Gradio）
- [ ] Arize Phoenix 可观测性接入
- [ ] DEV-PROMO-003/004/005（归因偏差 / 多门店 / 天气异常）
- [ ] TimesFM 零样本预测对比基线

---

## Stay in Touch

左边是**公众号**，更新项目动态与 AI-Native 供应链的实战内容；
右边是**个人微信**，交流、反馈、提 bug、商业合作都欢迎。

<table>
  <tr>
    <td align="center">
      <img src="imgs/gongzhonghao.jpg" alt="微信公众号" width="200"><br>
      <sub>微信公众号</sub>
    </td>
    <td align="center">
      <img src="imgs/kefu.png" alt="个人微信" width="200"><br>
      <sub>个人微信（交流 / 反馈）</sub>
    </td>
  </tr>
</table>

---

## Contributing

Issues and PRs welcome. Please do not commit any real customer data or API keys.

---

## License

MIT

---

## English Version

**retail-supply-chain-agent** is an AI-Native safety-stock agent for retail supply chains, focused on promotional scenarios. Built on **LangGraph 5-Expert + Critic** multi-agent architecture.

### What makes it different

Most supply chain forecasting tools are either pure ML (black box, no explanation) or pure rules (rigid, no learning). This project wires Claude API into the decision loop at four points:

1. **Intent parsing** — Claude `tool_use` extracts structured slots from natural language queries (store, SKU, discount rate, dates)
2. **Model routing** — Claude decides which forecast model family to use based on scenario features
3. **Quality scoring** — Claude LLM-as-Judge scores outputs on 6 dimensions (accuracy, completeness, compliance, executability, process rationality, business value)
4. **Attribution narrative** — Claude generates differentiated natural-language explanations of what drove the forecast

Every Claude call has a rule-based fallback, so the system runs offline too.

### Architecture

```
Natural language query
  → Layer1: Claude tool_use slot extraction
  → Layer2: LangGraph orchestration (Planner + 5 Experts + Critic)
      ForecastAgent  → Claude routes to LightGBM quantile regression
      SafetyStockAgent → Z-Score: SS = z × σ × √L
      WhatIfAgent    → elasticity-corrected scenario comparison
      ExplainAgent   → Claude-generated attribution narrative
      CriticAgent    → Claude LLM-as-Judge (6-dim rubric)
  → Layer3: LightGBM P25/P50/P75
  → Layer4: ActionBuilder (order qty + confidence tier)
  → Governance: HITLGate + AuditLogger + SqliteSaver
```

### Quick start

```bash
git clone https://github.com/CyAlcher/retail_supply_chain_agent.git
cd retail_supply_chain_agent
pip install -e . && pip install langgraph-checkpoint-sqlite
export ANTHROPIC_API_KEY=your_key

python data/promo_synthetic/gen_data.py
python demo/run_demo.py --case DEV-PROMO-001 --validate
python demo/run_demo.py --replay DEV-PROMO-001   # replay from SQLite
```

### Stack

`LangGraph 1.0` · `LightGBM 4.6` · `Pydantic v2` · `Claude API (Haiku)` · `SQLite checkpoint`

### Recommended GitHub Topics

`supply-chain` `retail` `ai-agent` `langgraph` `claude-api` `llm-as-judge` `demand-forecasting` `safety-stock` `multi-agent` `python` `lightgbm` `pydantic`
