"""合成促销历史数据生成器
促销期销量 = 基线 × (1 + 0.8×discount_rate) × 天气系数 × 周末系数 + 噪声
基线日销量 ~N(600, 80)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "promo_history.csv"

def generate(n_days: int = 180) -> pd.DataFrame:
    dates = pd.date_range("2025-09-01", periods=n_days, freq="D")
    baseline = RNG.normal(600, 80, n_days).clip(300, 1000)

    discount_rate = np.zeros(n_days)
    is_promo = np.zeros(n_days, dtype=bool)
    # 每 14 天安排一次 3 天促销
    for start in range(0, n_days, 14):
        end = min(start + 3, n_days)
        discount_rate[start:end] = RNG.choice([0.15, 0.20, 0.25, 0.30])
        is_promo[start:end] = True

    avg_temp = 28 + 6 * np.sin(np.linspace(0, 2 * np.pi, n_days)) + RNG.normal(0, 2, n_days)
    is_weekend = np.array([d.weekday() >= 5 for d in dates])

    weather_coef  = 1 + 0.015 * (avg_temp - 25).clip(-5, 10)
    weekend_coef  = np.where(is_weekend, 1.18, 1.0)
    uplift_coef   = 1 + 0.8 * discount_rate
    noise         = RNG.normal(0, 40, n_days)

    sales_qty = (baseline * uplift_coef * weather_coef * weekend_coef + noise).clip(0).round().astype(int)

    df = pd.DataFrame({
        "date":          dates.strftime("%Y-%m-%d"),
        "sku_id":        "SKU_COKE_330ML",
        "store_id":      "STORE_CRW_TH",
        "sales_qty":     sales_qty,
        "discount_rate": discount_rate.round(2),
        "is_promo":      is_promo.astype(int),
        "avg_temp":      avg_temp.round(1),
        "is_weekend":    is_weekend.astype(int),
        "baseline_qty":  baseline.round().astype(int),
    })
    return df

if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"生成 {len(df)} 条记录 → {OUT}")
    promo = df[df["is_promo"] == 1]
    print(f"促销天数: {len(promo)}  平均 uplift: {(promo['sales_qty']/promo['baseline_qty']).mean():.2f}x")
