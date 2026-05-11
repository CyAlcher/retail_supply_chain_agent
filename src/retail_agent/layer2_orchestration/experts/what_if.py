"""WhatIfAgent：促销力度对比推演（DEV-PROMO-002）
TODO(真实化): 用 Claude tool_use 做多方案推演决策
"""
from __future__ import annotations
from retail_agent.schemas import WhatIfResult, WhatIfScenario, PlannerState
from retail_agent.layer3_compute.forecast_engine.ml import tool as ml_tool
from .base import BaseExpertAgent

_COST_RATE = 0.60   # 成本占售价比例
_BASE_PRICE = 4.5   # 元/罐
# 促销弹性系数：折扣率每增加 1%，销量增加约 1.2%（基于合成数据拟合）
_ELASTICITY = 1.2


class WhatIfAgent(BaseExpertAgent):
    name = "what_if"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        promo = task.promo
        if not promo:
            return state

        base_dr = promo.discount_rate
        alt_dr = min(base_dr + 0.10, 0.50)

        scenarios = []
        for label, dr in [
            (f"方案A ({int(base_dr*100)}%折扣)", base_dr),
            (f"方案B ({int(alt_dr*100)}%折扣)", alt_dr),
        ]:
            pred = ml_tool.predict(
                discount_rate=dr,
                avg_temp=task.weather.avg_temp if task.weather else 28.0,
                is_weekend=task.calendar.is_weekend if task.calendar else False,
                duration_days=task.forecast_horizon_days,
            )
            # 用弹性系数修正 ML 预测，确保更高折扣产生可见的销量差异
            elasticity_boost = 1.0 + _ELASTICITY * (dr - base_dr)
            adjusted_p50 = round(pred["p50"] * elasticity_boost, 1)

            sell_price = _BASE_PRICE * (1 - dr)
            gross_profit = (sell_price - _BASE_PRICE * _COST_RATE) * adjusted_p50
            scenarios.append(WhatIfScenario(
                label=label,
                discount_rate=dr,
                forecast_p50=adjusted_p50,
                gross_profit=round(gross_profit, 1),
            ))

        best = max(scenarios, key=lambda s: s.gross_profit or 0)
        other = next(s for s in scenarios if s.label != best.label)
        sales_diff_pct = round(
            (scenarios[1].forecast_p50 - scenarios[0].forecast_p50)
            / max(scenarios[0].forecast_p50, 1) * 100, 1
        )
        profit_diff = round((best.gross_profit or 0) - (other.gross_profit or 0), 1)

        state.what_if_result = WhatIfResult(
            scenarios=scenarios,
            recommended=best.label,
            recommendation_reason=(
                f"方案B折扣提升10%，销量预计增加 {sales_diff_pct}%，"
                f"但毛利差额 ¥{profit_diff}，{best.label} 综合更优"
            ),
        )
        return state
