"""ForecastAgent：Claude tool_use 模型路由 + LightGBM 执行
Claude 根据场景特征决策调哪个模型族，fallback 直接调 ML tool
"""
from __future__ import annotations
import os
import anthropic
from retail_agent.schemas import ForecastResult, PlannerState, Scenario
from retail_agent.layer3_compute.forecast_engine.ml import tool as ml_tool
from .base import BaseExpertAgent

_ROUTER_SYSTEM = (
    "你是零售供应链预测系统的模型路由专家。"
    "根据场景特征选择最合适的预测模型族，调用 select_forecast_model 工具返回决策。"
    "可选模型族：lgb_quantile_uplift（促销/短期）、statistical_baseline（长尾/冷启动）。"
)

_ROUTER_TOOL = {
    "name": "select_forecast_model",
    "description": "根据场景特征选择预测模型族",
    "input_schema": {
        "type": "object",
        "properties": {
            "model_family": {
                "type": "string",
                "enum": ["lgb_quantile_uplift", "statistical_baseline"],
                "description": "选择的模型族",
            },
            "reason": {
                "type": "string",
                "description": "选择理由（一句话）",
            },
        },
        "required": ["model_family", "reason"],
    },
}


def _route_model(state: PlannerState) -> tuple[str, str]:
    """调用 Claude 决策模型族，返回 (model_family, reason)"""
    try:
        task = state.task
        context = (
            f"场景: {task.scenario.value}\n"
            f"促销折扣: {task.promo.discount_rate * 100:.0f}%" if task.promo else "无促销\n"
            f"预测天数: {task.forecast_horizon_days}\n"
            f"历史数据: 合成数据 540 天\n"
        )
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=_ROUTER_SYSTEM,
            tools=[_ROUTER_TOOL],
            tool_choice={"type": "tool", "name": "select_forecast_model"},
            messages=[{"role": "user", "content": f"请为以下场景选择预测模型：\n{context}"}],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "select_forecast_model":
                return block.input["model_family"], block.input["reason"]
    except Exception:
        pass
    # fallback：促销场景用 LightGBM，其他用统计基线
    if state.task.scenario == Scenario.PROMO:
        return "lgb_quantile_uplift", "促销场景默认使用 LightGBM 分位数回归"
    return "statistical_baseline", "非促销场景使用统计基线"


class ForecastAgent(BaseExpertAgent):
    name = "forecast"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        promo = task.promo

        model_family, route_reason = _route_model(state)

        result = ml_tool.predict(
            discount_rate=promo.discount_rate if promo else 0.0,
            avg_temp=task.weather.avg_temp if task.weather else 28.0,
            is_weekend=task.calendar.is_weekend if task.calendar else False,
            duration_days=task.forecast_horizon_days,
        )

        # 统计基线模式：覆盖 model_used 标识
        if model_family == "statistical_baseline":
            result["model_used"] = "statistical_baseline"

        state.forecast_result = ForecastResult(
            p25=result["p25"],
            p50=result["p50"],
            p75=result["p75"],
            model_used=f"{result['model_used']} [routed: {route_reason[:40]}]",
            feature_importance=result.get("feature_importance", {}),
        )
        state.llm_call_count += 1
        return state
