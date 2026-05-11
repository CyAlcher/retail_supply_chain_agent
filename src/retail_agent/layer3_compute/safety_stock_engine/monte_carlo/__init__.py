"""蒙特卡洛仿真安全库存引擎（A类高价值SKU）
对非正态分布需求更准确，输出置信区间而非点估计。
"""
from __future__ import annotations
import numpy as np


def calculate(
    demand_samples: list[float] | None = None,
    demand_mean: float = 600.0,
    demand_std: float = 80.0,
    service_level: float = 0.95,
    lead_time_days: int = 30,
    n_simulations: int = 10_000,
    rng_seed: int = 42,
) -> dict:
    """
    蒙特卡洛仿真安全库存。

    demand_samples: 历史日销量样本（有则用 bootstrap，无则用正态分布）
    返回 {safety_stock_p50, safety_stock_p95, coverage_days, method, ci_lower, ci_upper}
    """
    rng = np.random.default_rng(rng_seed)

    if demand_samples and len(demand_samples) >= 30:
        # Bootstrap：从历史样本有放回抽样
        samples = np.array(demand_samples)
        sim_demand = np.array([
            rng.choice(samples, size=lead_time_days, replace=True).sum()
            for _ in range(n_simulations)
        ])
        method = "bootstrap"
    else:
        # 正态分布仿真（fallback）
        sim_demand = rng.normal(
            loc=demand_mean * lead_time_days,
            scale=demand_std * (lead_time_days ** 0.5),
            size=n_simulations,
        )
        method = "normal_simulation"

    expected_demand = float(np.mean(sim_demand))
    ss_p50 = float(np.percentile(sim_demand, service_level * 100) - expected_demand)
    ss_p95 = float(np.percentile(sim_demand, 97.5) - expected_demand)
    ci_lower = float(np.percentile(sim_demand, 2.5))
    ci_upper = float(np.percentile(sim_demand, 97.5))

    ss_p50 = max(ss_p50, 0.0)
    ss_p95 = max(ss_p95, 0.0)

    avg_daily = demand_mean if demand_mean > 0 else (expected_demand / lead_time_days)
    coverage_days = ss_p50 / max(avg_daily, 1)

    return {
        "safety_stock_p50":  round(ss_p50, 1),
        "safety_stock_p95":  round(ss_p95, 1),
        "coverage_days":     round(coverage_days, 1),
        "service_level":     service_level,
        "ci_lower":          round(ci_lower, 1),
        "ci_upper":          round(ci_upper, 1),
        "expected_demand":   round(expected_demand, 1),
        "method":            method,
        "n_simulations":     n_simulations,
    }
