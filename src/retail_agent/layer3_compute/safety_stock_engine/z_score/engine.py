"""Z-Score 服务水平法安全库存引擎（纯数学，无 mock）
公式: SS = z × σ_d × √L
  z   = 正态分布分位数（服务水平对应）
  σ_d = 需求标准差（日）
  L   = 补货提前期（天）
"""
from __future__ import annotations
from scipy.stats import norm


def calculate(
    demand_std_daily: float,
    service_level: float = 0.95,
    lead_time_days: int = 3,
    forecast_horizon_days: int = 3,
    avg_daily_demand: float | None = None,
) -> dict:
    """
    返回 {safety_stock_units, coverage_days, service_level, z_score, demand_std}
    avg_daily_demand: 传入真实历史均值；不传则用 std×5 粗估
    """
    z = norm.ppf(service_level)
    safety_stock = z * demand_std_daily * (lead_time_days ** 0.5)
    safety_stock = max(safety_stock, 0.0)
    if avg_daily_demand is None:
        avg_daily_demand = demand_std_daily * 5  # 粗估兜底
    # coverage_days = 安全库存能覆盖多少天的需求波动
    coverage_days = safety_stock / max(avg_daily_demand, 1)

    return {
        "safety_stock_units": round(safety_stock, 1),
        "coverage_days":      round(coverage_days, 1),
        "service_level":      service_level,
        "z_score":            round(z, 4),
        "demand_std":         round(demand_std_daily, 1),
    }
