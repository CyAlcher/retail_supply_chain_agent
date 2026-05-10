"""ForecastAgent：调 LightGBM uplift 工具，输出 P25/P50/P75
TODO(真实化): 用 Claude tool_use 让 LLM 决策调哪个模型族
"""
from __future__ import annotations
from retail_agent.schemas import ForecastResult, PlannerState
from retail_agent.layer3_compute.forecast_engine.ml import tool as ml_tool
from .base import BaseExpertAgent


class ForecastAgent(BaseExpertAgent):
    name = "forecast"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        promo = task.promo

        result = ml_tool.predict(
            discount_rate=promo.discount_rate if promo else 0.0,
            avg_temp=task.weather.avg_temp if task.weather else 28.0,
            is_weekend=task.calendar.is_weekend if task.calendar else False,
            duration_days=task.forecast_horizon_days,
        )
        state.forecast_result = ForecastResult(
            p25=result["p25"],
            p50=result["p50"],
            p75=result["p75"],
            model_used=result["model_used"],
            feature_importance=result.get("feature_importance", {}),
        )
        state.llm_call_count += 0  # mock: 无 LLM 调用
        return state
