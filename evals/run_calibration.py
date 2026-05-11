#!/usr/bin/env python3
"""
Critic LLM-as-Judge 校准集验证脚本
用法：conda run -n agent python evals/run_calibration.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

import yaml
from retail_agent.schemas import (
    PlannerState, TaskContext, Scenario, ForecastResult,
    SafetyStockResult, WhatIfResult, WhatIfScenario,
)
from retail_agent.layer2_orchestration.experts.critic import CriticAgent

_CRITIC = CriticAgent()


def _build_state(sample: dict) -> PlannerState:
    ctx = sample["context"]
    task = TaskContext(
        task_id=sample["id"],
        sku_id="SKU_COKE_330ML",
        store_id="STORE_CRW_TH",
        scenario=Scenario.PROMO,
        raw_question="校准测试",
    )
    state = PlannerState(task=task)
    state.plan_experts = ctx.get("plan_experts", [])
    state.retry_count  = ctx.get("retry_count", 0)

    # 构造 audit_trail（visited_experts）
    for expert in ctx.get("visited_experts", []):
        state.audit_trail.append({"expert": expert, "status": "ok"})

    if ctx.get("forecast"):
        fc = ctx["forecast"]
        state.forecast_result = ForecastResult(
            p25=fc["p25"], p50=fc["p50"], p75=fc["p75"],
            model_used=fc.get("model", "lgb_quantile_uplift"),
        )

    if ctx.get("safety_stock"):
        ss = ctx["safety_stock"]
        state.safety_stock_result = SafetyStockResult(
            safety_stock_units=ss["units"],
            coverage_days=ss["coverage_days"],
            service_level=ss["service_level"],
            z_score=1.645,
            demand_std=80.0,
        )

    if ctx.get("what_if_scenarios", 0) > 0:
        state.what_if_result = WhatIfResult(
            scenarios=[
                WhatIfScenario(label="方案A", discount_rate=0.25, forecast_p50=2000, gross_profit=1500),
                WhatIfScenario(label="方案B", discount_rate=0.35, forecast_p50=2400, gross_profit=800),
            ],
            recommended="方案A",
            recommendation_reason="毛利更高",
        )

    return state


def run_calibration(yaml_path: str) -> dict:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    samples = data["calibration_samples"]
    results = []
    passed = 0

    print(f"\n{'='*60}")
    print(f"  Critic 校准集验证  ({len(samples)} 条样本)")
    print(f"{'='*60}")

    for s in samples:
        state = _build_state(s)
        quality, risks, uncertainty = _CRITIC.score(state)
        reflection = _CRITIC.decide(quality, risks, state.retry_count)

        score = quality.weighted_total
        lo, hi = s["expected_score_range"]
        score_ok = lo <= score <= hi
        refl_ok  = reflection.value == s["expected_reflection"]
        ok = score_ok and refl_ok

        if ok:
            passed += 1

        results.append({
            "id":       s["id"],
            "label":    s["label"],
            "score":    round(score, 3),
            "expected": f"[{lo},{hi}]",
            "score_ok": score_ok,
            "reflection":          reflection.value,
            "expected_reflection": s["expected_reflection"],
            "refl_ok":  refl_ok,
            "pass":     ok,
        })

        status = "✓" if ok else "✗"
        print(f"  {status} {s['id']:20s}  score={score:.2f} {s['expected_score_range']}  "
              f"refl={reflection.value}/{s['expected_reflection']}")

    pass_rate = passed / len(samples)
    print(f"\n  通过率: {passed}/{len(samples)} = {pass_rate*100:.0f}%")
    print(f"  {'✅ 校准通过（≥70%）' if pass_rate >= 0.70 else '❌ 校准未通过（<70%）'}")
    print(f"{'='*60}\n")

    return {"pass_rate": pass_rate, "passed": passed, "total": len(samples), "results": results}


if __name__ == "__main__":
    yaml_path = Path(__file__).parent / "critic_calibration.yaml"
    result = run_calibration(str(yaml_path))
    sys.exit(0 if result["pass_rate"] >= 0.70 else 1)
