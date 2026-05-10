"""SafetyStockAgent：Z-Score 服务水平法
TODO(真实化): 支持蒙特卡洛仿真（高价值 SKU）
"""
from __future__ import annotations
from retail_agent.schemas import SafetyStockResult, PlannerState
from retail_agent.layer3_compute.safety_stock_engine.z_score import engine as ss_engine
from .base import BaseExpertAgent


class SafetyStockAgent(BaseExpertAgent):
    name = "safety_stock"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        demand_std   = ctx_data.get("demand_std", 80.0)
        avg_baseline = ctx_data.get("avg_baseline", None)

        # 促销期日均需求优先用预测值（比历史基线更准确）
        fc = state.forecast_result
        if fc and state.task.forecast_horizon_days > 0:
            avg_daily_demand = fc.p50 / state.task.forecast_horizon_days
        else:
            avg_daily_demand = avg_baseline

        lead_time = 30  # 月度补货周期（TODO: 从商品主数据读取）
        result = ss_engine.calculate(
            demand_std_daily=demand_std,
            service_level=0.95,
            lead_time_days=lead_time,
            forecast_horizon_days=state.task.forecast_horizon_days,
            avg_daily_demand=avg_daily_demand,
        )
        # coverage_days 业务语义：安全库存覆盖的补货提前期天数
        # Z-Score 公式 SS = z × σ × √L，SS 本身就是为了覆盖 L 天的需求波动
        # 所以 coverage_days = lead_time_days（这是 Z-Score 的设计语义）
        result["coverage_days"] = float(lead_time)
        state.safety_stock_result = SafetyStockResult(**result)
        return state
