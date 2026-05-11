#!/usr/bin/env python3
"""
命令行演示入口
用法:
  conda run -n agent python demo/run_demo.py --case DEV-PROMO-001
  conda run -n agent python demo/run_demo.py --case DEV-PROMO-002
  conda run -n agent python demo/run_demo.py --case DEV-PROMO-001 --validate
  conda run -n agent python demo/run_demo.py --show-todos
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path

# 把 src 加入 path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from retail_agent.schemas import TaskContext, PromoFactor, WeatherFactor, CalendarFactor, Scenario
from retail_agent.layer1_perception.entry.entrypoint import parse as parse_question
from retail_agent.layer2_orchestration.planner.planner import run as run_graph

# ── 预设 case ─────────────────────────────────────────────────────────────────

CASES = {
    "DEV-PROMO-001": {
        "question": (
            "华润万家天河店，可口可乐 330ml 罐装，"
            "下周六（2026-06-13）开始做 25% 折扣促销 3 天，"
            "帮我预测销量并给出备货建议。"
        ),
        "overrides": {
            "task_id": "DEV-PROMO-001",
            "weather": WeatherFactor(avg_temp=32.0, rain_prob=0.3),
            "calendar": CalendarFactor(is_weekend=True),
        },
        # 硬断言（--validate 时检查）
        "assertions": {
            "p50_range": (1800, 2600),
            "p25_p75_width_max": 800,
            "coverage_days_min": 10,
            "action_type": "下单",
            "confidence_not_reject": True,
        },
    },
    "DEV-PROMO-002": {
        "question": (
            "同一家门店，同款可口可乐 330ml，"
            "如果把折扣从 20% 提到 30%，销量和毛利分别会怎么变化？"
            "哪个方案对门店更划算？"
        ),
        "overrides": {
            "task_id": "DEV-PROMO-002",
            "promo": PromoFactor(
                discount_rate=0.20,
                start_date="2026-06-13",
                end_date="2026-06-15",
                duration_days=3,
            ),
            "weather": WeatherFactor(avg_temp=30.0),
            "calendar": CalendarFactor(is_weekend=False),
        },
        "assertions": {
            "what_if_has_two_scenarios": True,
            "recommended_not_empty": True,
        },
    },
}

TODOS = """
╔══════════════════════════════════════════════════════════════════╗
║              Hardcode & AI-Native TODO 清单                      ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-01] 意图解析                                               ║
║   现状: 关键词规则匹配 (entrypoint.py)                           ║
║   TODO : Claude API tool_use slot-filling                        ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-02] ForecastAgent LLM 决策                                 ║
║   现状: 直接调 ML tool，跳过 LLM 路由决策                        ║
║   TODO : Claude tool_use 让 LLM 决策调哪个模型族                 ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-03] CriticAgent 评分                                       ║
║   现状: 规则判断（WAPE 阈值 + 区间宽度）                         ║
║   TODO : Claude LLM-as-Judge + 结构化 Rubric + 校准集            ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-04] ExplainAgent 叙述                                      ║
║   现状: 模板字符串拼接                                           ║
║   TODO : Claude 生成自然语言归因叙述                             ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-05] HITLGate                                               ║
║   现状: 自动通过（演示不阻塞）                                   ║
║   TODO : 真实审核界面（Web/IM），等待人工确认                    ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-06] LangGraph Checkpointer                                 ║
║   现状: MemorySaver（内存，进程退出即丢失）                      ║
║   TODO : PostgresSaver（持久化，支持决策回放）                   ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-07] 训练数据                                               ║
║   现状: 合成数据（统计特征真实，非真实 KA 客户数据）             ║
║   TODO : 接入真实 KA 客户历史销量 Feature Store                  ║
╠══════════════════════════════════════════════════════════════════╣
║ [MOCK-08] 可观测性                                               ║
║   现状: 终端打印                                                 ║
║   TODO : Arize Phoenix 自托管 + OpenTelemetry trace              ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── 打印函数 ──────────────────────────────────────────────────────────────────

def _sep(char="─", width=64):
    print(char * width)

def _print_result(state, case_id: str) -> None:
    fc = state.forecast_result
    ss = state.safety_stock_result
    wi = state.what_if_result
    ex = state.explain_result
    cv = state.critic_verdict
    ac = state.action

    _sep("═")
    print(f"  零售供应链 AI-Native Agent  |  {case_id}")
    _sep("═")

    print(f"\n[任务]  {state.task.raw_question}\n")
    _sep()

    # Planner
    print(f"[Planner]   意图识别 → 促销预测+补货建议"
          f"  场景路由 → {state.task.scenario.value}")
    print(f"            Expert 计划: {state.plan_experts}")

    # Critic ①
    if cv:
        plan_ok = "✓ 通过" if cv.plan_ok else "✗ 失败"
        print(f"\n[Critic①]  计划合理性校验 → {plan_ok}"
              f"  Expert: {state.plan_experts}")

    # Forecast
    if fc:
        print(f"\n[Forecast]  ML预测工具调用 → "
              f"P25={fc.p25:.0f}  P50={fc.p50:.0f}  P75={fc.p75:.0f}"
              f"  模型={fc.model_used}")
        if fc.feature_importance:
            top = sorted(fc.feature_importance.items(), key=lambda x: -x[1])[:3]
            print(f"            特征重要性: {dict(top)}")

    # SafetyStock
    if ss:
        print(f"\n[SafetyStock] Z-Score计算 → "
              f"安全库存={ss.safety_stock_units:.0f}件"
              f"  覆盖天数={ss.coverage_days:.1f}"
              f"  服务水平={ss.service_level*100:.0f}%"
              f"  z={ss.z_score:.3f}")

    # WhatIf（PROMO-002）
    if wi:
        print(f"\n[WhatIf]    对比推演:")
        for s in wi.scenarios:
            print(f"            {s.label}: P50={s.forecast_p50:.0f}件  毛利=¥{s.gross_profit:.0f}")
        print(f"            推荐 → {wi.recommended}")
        print(f"            理由 → {wi.recommendation_reason}")

    # Explain
    if ex:
        print(f"\n[Explain]   归因:")
        for f in ex.key_drivers:
            sign = "+" if f.contribution_pct >= 0 else ""
            print(f"            {f.factor}: {sign}{f.contribution_pct:.1f}%  ({f.data_source})")
        print(f"            叙述: {ex.narrative}")
        print(f"            下一步: {ex.next_actions}")

    # Critic ②
    if cv and cv.quality:
        q = cv.quality
        print(f"\n[Critic②]  质量评分 → {q.weighted_total:.2f}"
              f"  (准确={q.accuracy:.2f} 完备={q.completeness:.2f}"
              f" 合规={q.compliance:.2f} 可执行={q.executability:.2f})")
        print(f"            风险: {cv.risks if cv.risks else '无'}")
        print(f"            决策: {cv.reflection.value}")

    # Action
    if ac:
        print(f"\n[Action]    建议动作={ac.action_type}"
              f"  数量={ac.quantity:.0f}件"
              f"  置信度={ac.confidence_tier.value}")
        print(f"            理由: {ac.rationale}")

    _sep()


def _validate(state, assertions: dict) -> bool:
    fc = state.forecast_result
    ss = state.safety_stock_result
    wi = state.what_if_result
    ac = state.action
    passed = True

    print("\n[Validate] 硬断言检查:")

    if "p50_range" in assertions and fc:
        lo, hi = assertions["p50_range"]
        ok = lo <= fc.p50 <= hi
        print(f"  P50 ∈ [{lo}, {hi}]: {fc.p50:.0f}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "p25_p75_width_max" in assertions and fc:
        w = fc.p75 - fc.p25
        ok = w <= assertions["p25_p75_width_max"]
        print(f"  区间宽度 ≤ {assertions['p25_p75_width_max']}: {w:.0f}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "coverage_days_min" in assertions and ss:
        ok = ss.coverage_days >= assertions["coverage_days_min"]
        print(f"  覆盖天数 ≥ {assertions['coverage_days_min']}: {ss.coverage_days:.1f}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "action_type" in assertions and ac:
        ok = ac.action_type == assertions["action_type"]
        print(f"  动作类型={assertions['action_type']}: {ac.action_type}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "confidence_not_reject" in assertions and ac:
        from retail_agent.schemas import ConfidenceTier
        ok = ac.confidence_tier != ConfidenceTier.REJECT
        print(f"  置信度≠拒绝: {ac.confidence_tier.value}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "what_if_has_two_scenarios" in assertions and wi:
        ok = len(wi.scenarios) == 2
        print(f"  WhatIf 有2个方案: {len(wi.scenarios)}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    if "recommended_not_empty" in assertions and wi:
        ok = bool(wi.recommended)
        print(f"  推荐方案非空: {wi.recommended!r}  {'✓' if ok else '✗ FAIL'}")
        passed = passed and ok

    print(f"\n  总结: {'全部通过 ✓' if passed else '存在失败项 ✗'}")
    return passed


# ── 主函数 ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="零售供应链 AI-Native Agent 演示")
    ap.add_argument("--case", choices=list(CASES.keys()), default="DEV-PROMO-001")
    ap.add_argument("--validate", action="store_true", help="跑完后检查硬断言")
    ap.add_argument("--show-todos", action="store_true", help="打印 Hardcode/TODO 清单")
    ap.add_argument("--replay", metavar="TASK_ID", help="从 SQLite checkpoint 重放历史决策")
    args = ap.parse_args()

    if args.show_todos:
        print(TODOS)
        return

    if args.replay:
        from retail_agent.layer2_orchestration.planner.planner import replay
        print(f"\n正在重放 {args.replay} 的历史决策...")
        final_state = replay(args.replay)
        if final_state is None:
            print(f"未找到 {args.replay} 的历史记录，请先运行该 case。")
            sys.exit(1)
        _print_result(final_state, args.replay)
        return

    if args.show_todos:
        print(TODOS)
        return

    cfg = CASES[args.case]
    print(f"\n正在运行 {args.case}，解析问题并构建上下文...")

    # 解析自然语言 → TaskContext
    ctx = parse_question(cfg["question"], task_id=cfg["overrides"].get("task_id", args.case))

    # 应用 case 级覆盖（确保演示数据精确）
    for k, v in cfg["overrides"].items():
        if k != "task_id":
            setattr(ctx, k, v)
    ctx.scenario = Scenario.PROMO  # 两个 case 都是促销场景

    from retail_agent.schemas import PlannerState
    state = PlannerState(task=ctx)

    # 运行 LangGraph 图
    final_state = run_graph(state)

    # 打印结果
    _print_result(final_state, args.case)

    # 可选验证
    if args.validate:
        ok = _validate(final_state, cfg.get("assertions", {}))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
