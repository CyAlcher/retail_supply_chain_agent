"""ExplainAgent：规则模板归因 + 自然语言叙述
TODO(真实化): 用 Claude 生成自然语言叙述，SHAP 做特征归因
"""
from __future__ import annotations
from retail_agent.schemas import ExplainResult, AttributionFactor, PlannerState
from .base import BaseExpertAgent


class ExplainAgent(BaseExpertAgent):
    name = "explain"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        fc = state.forecast_result
        promo = task.promo

        factors: list[AttributionFactor] = []
        remaining = 100.0

        # 促销 uplift 因子
        if promo and promo.discount_rate > 0:
            uplift_pct = round(promo.discount_rate * 0.8 * 100, 1)
            factors.append(AttributionFactor(
                factor="历史同力度促销 uplift",
                contribution_pct=uplift_pct,
                data_source="合成历史促销数据（相似折扣率样本）",
            ))
            remaining -= uplift_pct

        # 天气因子
        if task.weather and task.weather.avg_temp:
            temp_adj = round((task.weather.avg_temp - 25) * 0.015 * 100, 1)
            temp_adj = max(min(temp_adj, remaining * 0.4), -remaining * 0.4)
            factors.append(AttributionFactor(
                factor="天气调整因子",
                contribution_pct=round(temp_adj, 1),
                data_source=f"气温 {task.weather.avg_temp}℃",
            ))
            remaining -= abs(temp_adj)

        # 周末因子
        if task.calendar and task.calendar.is_weekend:
            weekend_pct = round(min(18.0, remaining * 0.5), 1)
            factors.append(AttributionFactor(
                factor="周末效应",
                contribution_pct=weekend_pct,
                data_source="历史周末 vs 工作日销量对比",
            ))
            remaining -= weekend_pct

        # 基线
        if remaining > 0:
            factors.append(AttributionFactor(
                factor="基线销量",
                contribution_pct=round(remaining, 1),
                data_source="近 90 天日均销量",
            ))

        # 叙述（TODO: 替换为 Claude 生成）
        main = factors[0].factor if factors else "基线销量"
        narrative = (
            f"本次预测中位数 {fc.p50 if fc else 'N/A'} 件，"
            f"主要驱动因子为「{main}」。"
            f"建议关注促销陈列到位率，确保 uplift 兑现。"
        )

        state.explain_result = ExplainResult(
            key_drivers=factors,
            narrative=narrative,
            next_actions=["确认促销陈列到位", "提前 3 天下单备货", "监控首日销量偏差"],
        )
        return state
