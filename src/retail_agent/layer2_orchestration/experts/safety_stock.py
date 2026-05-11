"""SafetyStockAgent：Z-Score（默认）+ 蒙特卡洛仿真（A类高价值SKU）
A类SKU判断：sku_id 包含 'A_' 前缀，或 avg_daily_demand > 500
"""
from __future__ import annotations
from retail_agent.schemas import SafetyStockResult, PlannerState
from retail_agent.layer3_compute.safety_stock_engine.z_score import engine as ss_engine
from retail_agent.layer3_compute.safety_stock_engine.monte_carlo import calculate as mc_calculate
from .base import BaseExpertAgent

_A_CLASS_THRESHOLD = 500  # 日均需求超过此值视为 A 类 SKU


def _is_a_class(sku_id: str, avg_daily_demand: float) -> bool:
    return sku_id.startswith("A_") or avg_daily_demand > _A_CLASS_THRESHOLD


class SafetyStockAgent(BaseExpertAgent):
    name = "safety_stock"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        demand_std   = ctx_data.get("demand_std", 80.0)
        avg_baseline = ctx_data.get("avg_baseline", None)
        history_df   = ctx_data.get("history_df", None)

        fc = state.forecast_result
        if fc and state.task.forecast_horizon_days > 0:
            avg_daily_demand = fc.p50 / state.task.forecast_horizon_days
        else:
            avg_daily_demand = avg_baseline or demand_std * 5

        lead_time = 30

        if _is_a_class(state.task.sku_id, avg_daily_demand):
            # 蒙特卡洛仿真（A类SKU）
            demand_samples = (
                list(history_df["sales_qty"].values)
                if history_df is not None and len(history_df) >= 30
                else None
            )
            mc = mc_calculate(
                demand_samples=demand_samples,
                demand_mean=avg_daily_demand,
                demand_std=demand_std,
                service_level=0.95,
                lead_time_days=lead_time,
            )
            result = {
                "safety_stock_units": mc["safety_stock_p50"],
                "coverage_days":      float(lead_time),
                "service_level":      mc["service_level"],
                "z_score":            1.645,
                "demand_std":         demand_std,
            }
        else:
            # Z-Score（默认）
            result = ss_engine.calculate(
                demand_std_daily=demand_std,
                service_level=0.95,
                lead_time_days=lead_time,
                forecast_horizon_days=state.task.forecast_horizon_days,
                avg_daily_demand=avg_daily_demand,
            )
            result["coverage_days"] = float(lead_time)

        state.safety_stock_result = SafetyStockResult(**result)
        return state
