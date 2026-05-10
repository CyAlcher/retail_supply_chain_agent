"""WhatIfAgent：促销力度对比推演（DEV-PROMO-002）
TODO(真实化): 用 Claude tool_use 做多方案推演决策
"""
from __future__ import annotations
from retail_agent.schemas import WhatIfResult, WhatIfScenario, PlannerState
from retail_agent.layer3_compute.forecast_engine.ml import tool as ml_tool
from .base import BaseExpertAgent

# 毛利率假设（TODO: 从商品主数据读取）
_COST_RATE = 0.60   # 成本占售价比例
_BASE_PRICE = 4.5   # 元/罐


class WhatIfAgent(BaseExpertAgent):
    name = "what_if"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        promo = task.promo
        if not promo:
            return state  # 非促销场景跳过

        # 对比方案：当前折扣 vs 当前折扣 ±10%
        base_dr = promo.discount_rate
        candidates = {
            f"方案A ({int(base_dr*100)}%折扣)": base_dr,
            f"方案B ({int((base_dr+0.10)*100)}%折扣)": min(base_dr + 0.10, 0.50),
        }

        scenarios = []
        for label, dr in candidates.items():
            pred = ml_tool.predict(
                discount_rate=dr,
                avg_temp=task.weather.avg_temp if task.weather else 28.0,
                is_weekend=task.calendar.is_weekend if task.calendar else False,
                duration_days=task.forecast_horizon_days,
            )
            sell_price = _BASE_PRICE * (1 - dr)
            gross_profit = (sell_price - _BASE_PRICE * _COST_RATE) * pred["p50"]
            scenarios.append(WhatIfScenario(
                label=label,
                discount_rate=dr,
                forecast_p50=pred["p50"],
                gross_profit=round(gross_profit, 1),
            ))

        # 推荐毛利更高的方案
        best = max(scenarios, key=lambda s: s.gross_profit or 0)
        other = [s for s in scenarios if s.label != best.label][0]
        diff = round((best.gross_profit or 0) - (other.gross_profit or 0), 1)

        state.what_if_result = WhatIfResult(
            scenarios=scenarios,
            recommended=best.label,
            recommendation_reason=(
                f"毛利差额 ¥{diff}，促销弹性约 "
                f"{round((best.forecast_p50 - other.forecast_p50) / max(other.forecast_p50,1) * 100, 1)}%"
            ),
        )
        return state
