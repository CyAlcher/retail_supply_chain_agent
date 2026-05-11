# retail-supply-chain-agent 能力评测报告

> 评测视角：2026年5月 资深 AI-Native Agent 产品评测专家  
> 评测基准：Anthropic Building Effective Agents（2024-12）、AI-Native 四要素框架、DMALL 落地规划文档  
> 评测版本：v0.2.0（commit d9b94e3）  
> 评测日期：2026-05-12  
> 评测方法：真实 demo 运行 + 代码审查 + 业界最佳实践对标

---

## 一、评测总览

| 维度 | 得分 | 等级 |
|---|---|---|
| 自然语言理解与意图解析 | 8.5 / 10 | ★★★★☆ |
| 多 Agent 编排与协同 | 7.5 / 10 | ★★★★☆ |
| 预测计算能力 | 7.0 / 10 | ★★★☆☆ |
| 质量评估（Critic / LLM-as-Judge） | 7.5 / 10 | ★★★★☆ |
| 可解释性与归因叙述 | 8.0 / 10 | ★★★★☆ |
| 治理与可靠性 | 6.5 / 10 | ★★★☆☆ |
| AI-Native 四要素完整度 | 7.5 / 10 | ★★★★☆ |
| **综合评分** | **7.5 / 10** | **★★★★☆** |

**一句话结论**：架构骨架扎实，Claude API 四路接入真实可用，促销场景端到端闭环跑通；主要短板在链路可靠性（audit_trail 累积 bug 已修复但暴露了状态管理脆弱性）、预测精度（合成数据局限）、以及 HITL 和可观测性尚未真正演示。

---

## 二、各模块能力评测

### 2.1 自然语言理解与意图解析（8.5 / 10）

**测试输入**（零售业务通俗语言）：
```
华润万家天河店，可口可乐 330ml 罐装，
下周六（2026-06-13）开始做 25% 折扣促销 3 天，
帮我预测销量并给出备货建议。
```

**实测输出**：
```
sku_id: SKU_COKE_330ML
store_id: STORE_CRW_TH
discount_rate: 0.25
scenario: Scenario.PROMO
is_weekend: True
```

**评分依据**：
- ✅ Claude tool_use 正确提取折扣率（25%→0.25）、门店映射（华润万家天河店→STORE_CRW_TH）、周末识别（周六→True）
- ✅ 有规则 fallback，API 失败不崩溃
- ✅ 日期推算（start_date + duration → end_date）逻辑正确
- ⚠️ SKU 映射依赖硬编码字典，"可口可乐 330ml 罐装"以外的商品无法识别
- ⚠️ 多门店场景（DEV-PROMO-004）仍用同一门店数据，未真正区分门店

**业界对标**：达到 Shopify Sidekick 冷启动阶段水平，槽位提取准确，但商品/门店知识库覆盖不足。

---

### 2.2 多 Agent 编排与协同（7.5 / 10）

**LangGraph 图结构**：
```
START → plan → critic_check_plan → forecast → safety_stock
      → [what_if] → critic_score
        ├─ accept → explain → action_builder → hitl → audit → END
        ├─ retry(≤2) → forecast
        └─ escalate → hitl → audit → END
```

**实测表现**（DEV-PROMO-001）：
```
[Planner]   Expert 计划: ['forecast', 'safety_stock', 'explain', 'what_if']
[Critic①]  计划合理性校验 → ✓ 通过
[Forecast]  P25=2188  P50=2375  P75=2538
[SafetyStock] 安全库存=1117件
[WhatIf]    方案A vs 方案B → 推荐方案A
[Explain]   4个归因因子 + Claude 生成叙述
[Critic②]  质量评分=0.90  决策=accept
[Action]    下单 3493件
```

**评分依据**：
- ✅ 5 Expert 全部触发，符合 Anthropic Orchestrator-Workers 模式
- ✅ Critic 双检（计划前置 + 结果后置）真实运行
- ✅ retry/escalate 条件路由逻辑正确
- ✅ Claude 模型路由（ForecastAgent）真实决策模型族
- ⚠️ `llm_calls=2`（仅 ForecastAgent 路由 + ExplainAgent），CriticAgent 的 LLM-as-Judge 在本次运行中走了 fallback 规则路径（可能是 API 响应慢）
- ❌ 多门店并行（T-12 LangGraph Send API）未实现，DEV-PROMO-004 是串行单门店

**业界对标**：达到 AWS Bedrock Multi-Agent 基础水平，图结构完整，但并行能力缺失。

---

### 2.3 预测计算能力（7.0 / 10）

**实测数据**（DEV-PROMO-001，25% 折扣，32℃，周末）：
```
LightGBM P25=2188  P50=2375  P75=2538  区间宽度=350
统计基线 P50=2812  vs LGB偏差=15.5%
安全库存=1117件（Z-Score，SL=95%，L=30天）
```

**评分依据**：
- ✅ LightGBM 分位数回归（P25/P50/P75）真实训练，非 mock
- ✅ Z-Score 安全库存公式 `SS = z × σ × √L` 数学正确
- ✅ 统计基线对比（LGB vs 历史均值法）有实际偏差数字
- ✅ 蒙特卡洛仿真（A类SKU）已实现，bootstrap + 正态仿真双路径
- ⚠️ 训练数据为合成数据（540天），LGB 特征重要性 avg_temp 占 78.9% 偏高，说明模型对温度过拟合
- ⚠️ LGB vs 基线偏差 15.5% 无法判断谁更准（无真实标签），仅作参考
- ❌ TimesFM / Foundation TSM 未接入（环境无 torch）
- ❌ 动态集成（多模型投票）未实现

**业界对标**：达到 Blue Yonder Cognitive 早期版本水平，ML 预测可用，但缺少 Foundation TSM 和动态集成。

---

### 2.4 质量评估（Critic / LLM-as-Judge）（7.5 / 10）

**实测输出**（DEV-PROMO-001）：
```
[Critic②]  质量评分=0.90
  准确=0.82  完备=0.95  合规=1.00  可执行=0.90
  风险: ['预测区间宽度349.9单位占P50的14.7%...', '安全库存30天覆盖天数需验证...', ...]
  决策: accept
```

**六维评分实测**（DEV-PROMO-005，暴雨橙色预警）：
```
风险: [..., '气象预警: 暴雨橙色']  ← 强制注入，不依赖 LLM
决策: accept（置信度=需要复核）
```

**评分依据**：
- ✅ Claude LLM-as-Judge 六维 Rubric 结构化评分，有 tool_use 约束
- ✅ 气象预警强制注入机制，确保高风险场景不被 LLM 遗漏
- ✅ fallback 规则评分，API 失败不影响流程
- ✅ retry/escalate 反思决策逻辑正确
- ⚠️ Critic 风险列表包含"action字段为null"——这是时序问题（action 在 Critic 之后生成），属误报，降低了风险列表可信度
- ⚠️ 校准集缺失：无法验证 LLM 评分与人工评分的一致性（Rubric 校准是 LLM-as-Judge 的核心挑战）
- ❌ 无 WAPE/MAPE 真实值（无真实标签），准确性评分依赖区间宽度代理指标

**业界对标**：达到 Anthropic Evaluator-Optimizer Pattern 基础实现，但校准集缺失是生产化的主要障碍。

---

### 2.5 可解释性与归因叙述（8.0 / 10）

**实测输出**（DEV-PROMO-001，Claude 生成）：
```
叙述: 本次预测2375件主要由三个因素驱动：近90天日均销量基线贡献了51.5%的增长，
25%促销折扣相比历史同力度促销带动20%的销量提升，叠加32℃高温天气的10.5%拉动
和周末效应的18%增幅，综合形成本次预测。
```

**评分依据**：
- ✅ Claude 生成的叙述有具体数字支撑，不是套话
- ✅ 四个归因因子（促销 uplift / 天气 / 周末 / 基线）覆盖主要驱动
- ✅ 不同 case 叙述有差异（DEV-PROMO-003 偏差归因 vs DEV-PROMO-001 正向预测）
- ✅ 有 fallback 模板，API 失败不影响输出
- ⚠️ 归因系数（20% uplift、10.5% 天气）是规则计算而非 SHAP 值，不是真正的特征归因
- ⚠️ 叙述长度固定（2-3句），缺少对异常场景的深度分析
- ❌ SHAP 特征归因未实现（TODO 中提及）

**业界对标**：超过大多数传统预测系统的解释能力，接近 RELEX 的自然语言解释水平。

---

### 2.6 治理与可靠性（6.5 / 10）

**实测表现**：

| 治理模块 | 状态 | 实测结果 |
|---|---|---|
| HITLGate | ✅ 已实现 | Gradio UI 代码完整，HITL_AUTO_APPROVE=1 跳过 |
| AuditLogger | ✅ 已实现 | 终端打印正常，Phoenix trace 需 PHOENIX_ENABLED=1 |
| SQLiteSaver | ✅ 已实现 | --replay 功能验证通过 |
| VersionRegistry | ✅ 已实现 | register/rollback/list-versions 功能正常 |
| 链路可靠性 | ⚠️ 有缺陷 | audit_trail 累积 bug（已修复），暴露状态管理脆弱性 |
| 端到端成功率 | ⚠️ 未测量 | 5个 case 全部通过，但无压测数据 |

**评分依据**：
- ✅ 四大治理模块（HITL/Audit/Replay/Version）骨架完整
- ✅ audit_trail 累积 bug 已修复（delete_thread 前置清理）
- ⚠️ HITLGate Gradio UI 未真实演示（HITL_AUTO_APPROVE=1 默认跳过）
- ⚠️ Phoenix 可观测性未真实演示（需手动启动 Phoenix server）
- ⚠️ 无熔断/超时机制，LLM API 超时会导致整个 Agent 链路挂起
- ❌ 端到端 SLA 未测量（目标：单步≥95%，端到端≥85%）
- ❌ 无 Red-Team / 对抗测试

**业界对标**：达到 MVP 骨架水平，距离生产级治理（RELEX/Blue Yonder 标准）还有 2-3 个迭代。

---

## 三、AI-Native 四要素完整度评估

| 要素 | 目标 | 当前实现 | 完成度 |
|---|---|---|---|
| **自主决策** | Planner + 5 Expert 全链路自主 | 5 Expert 全触发，Claude 路由决策 | 80% |
| **质量评估** | Critic LLM-as-Judge + 校准集 | 六维评分实现，校准集缺失 | 70% |
| **持续进化** | 销量回传 → 模型迭代 → 版本管理 | VersionRegistry 骨架，学习闭环未实现 | 40% |
| **预判未来** | What-if + 未来因子注入 | What-if 弹性修正，促销/天气/日历因子 | 75% |

**四要素综合完成度：66%**

---

## 四、与 DMALL 落地规划 Gap 对比

对照 `AI-Native-落地规划.md` 的 MVP 必做 13 项：

| 项目 | 规划要求 | 当前状态 |
|---|---|---|
| 统一任务入口 | Claude tool_use 槽位解析 | ✅ 已实现 |
| 上下文聚合 | 历史数据 + 未来因子 | ✅ 已实现（合成数据） |
| Planner Agent | LangGraph StateGraph | ✅ 已实现 |
| Forecast Agent | LightGBM + 模型路由 | ✅ 已实现 |
| Safety Stock Agent | Z-Score + 蒙特卡洛 | ✅ 已实现 |
| What-if Agent | 促销力度敏感性 | ✅ 已实现（弹性修正） |
| Explain Agent | Claude 自然语言归因 | ✅ 已实现 |
| Critic Agent | LLM-as-Judge 六维评分 | ✅ 已实现（校准集缺失） |
| HITL | 低置信场景人工审核 | ⚠️ 代码完整，未真实演示 |
| 审计日志 | 全链路结构化日志 | ✅ 已实现（Phoenix 待启动） |
| 决策回放 | SQLite checkpoint + --replay | ✅ 已实现 |
| 版本回滚 | VersionRegistry + --rollback | ✅ 已实现（最小骨架） |
| Eval-as-Code | 5个 case + 硬断言 | ✅ 已实现 |

**MVP 13项完成度：12/13（92%）**，HITL 真实演示是唯一未完成项。

---

## 五、关键风险与改进建议

### 风险1（高）：预测精度无法验证
- **现状**：合成数据训练，无真实标签，MAPE 无法计算
- **影响**：无法向业务方证明预测价值
- **建议**：接入至少 1 个真实 KA 客户历史数据集，计算真实 MAPE

### 风险2（中）：LLM-as-Judge 校准缺失
- **现状**：Critic 评分无人工标注对照集
- **影响**：评分可信度存疑，高管汇报时被追问
- **建议**：构建 20-30 条人工标注的"好/中/差"预测结果，校准 Rubric

### 风险3（中）：链路可靠性未压测
- **现状**：5个 case 通过，无并发/超时/API 失败测试
- **影响**：生产环境可能出现挂起或状态污染
- **建议**：增加 API 超时熔断（30s），补充异常路径测试

### 风险4（低）：HITL 未真实演示
- **现状**：HITL_AUTO_APPROVE=1 默认跳过 Gradio UI
- **影响**：演示时无法展示人工介入价值
- **建议**：录制一次真实 HITL 审核的 GIF，放入 README

---

## 六、综合评级

```
┌─────────────────────────────────────────────────────┐
│  retail-supply-chain-agent  v0.2.0  综合评级         │
│                                                     │
│  架构完整度    ████████░░  80%                       │
│  Claude接入    █████████░  90%                       │
│  业务可用性    ███████░░░  70%                       │
│  生产就绪度    █████░░░░░  50%                       │
│                                                     │
│  综合评分：7.5 / 10  ★★★★☆                          │
│                                                     │
│  定位：面试/演示级 MVP，架构叙事完整，               │
│        距生产部署还需 2-3 个迭代                     │
└─────────────────────────────────────────────────────┘
```

**适合用于**：技术面试、产品委员会汇报、AI-Native 架构演示  
**不适合用于**：直接生产部署、真实 KA 客户数据处理

---

## 七、下一步优先行动（按 ROI 排序）

1. **接入真实数据**（1周）：哪怕 1 个 SKU 的 90 天真实销量，计算真实 MAPE，让预测价值可量化
2. **构建 Critic 校准集**（3天）：20 条人工标注样本，验证 LLM-as-Judge 评分一致性
3. **录制 HITL 演示 GIF**（1天）：关闭 HITL_AUTO_APPROVE，录制一次完整的人工审核流程
4. **补充 API 熔断**（半天）：在 entrypoint/critic/explain 的 LLM 调用加 30s 超时
5. **启动 Phoenix 可观测性**（半天）：`PHOENIX_ENABLED=1` 跑一次 demo，截图放 README
