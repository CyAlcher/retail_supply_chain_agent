"""合成促销历史数据生成器
促销期销量 = 基线 × (1 + 0.8×discount_rate) × 天气系数 × 周末系数 + 噪声
基线日销量：天河店 ~N(600,80)，越秀店 ~N(510,70)，海珠店 ~N(450,65)
"""
import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)
OUT = Path(__file__).parent / "promo_history.csv"

# 门店基线参数：(均值, 标准差, 基线倍数)
_STORES = {
    "STORE_CRW_TH": (600, 80,  1.00),   # 天河店：旗舰店，基线最高
    "STORE_CRW_YX": (510, 70,  0.85),   # 越秀店：中型店
    "STORE_CRW_HZ": (450, 65,  0.75),   # 海珠店：社区店，基线最低
}


def _generate_store(store_id: str, n_days: int, rng: np.random.Generator) -> pd.DataFrame:
    mean, std, _ = _STORES[store_id]
    dates    = pd.date_range("2025-09-01", periods=n_days, freq="D")
    baseline = rng.normal(mean, std, n_days).clip(mean * 0.5, mean * 1.7)

    discount_rate = np.zeros(n_days)
    is_promo      = np.zeros(n_days, dtype=bool)
    for start in range(0, n_days, 14):
        end = min(start + 3, n_days)
        discount_rate[start:end] = rng.choice([0.15, 0.20, 0.25, 0.30])
        is_promo[start:end]      = True

    avg_temp     = 28 + 6 * np.sin(np.linspace(0, 2 * np.pi, n_days)) + rng.normal(0, 2, n_days)
    is_weekend   = np.array([d.weekday() >= 5 for d in dates])

    weather_coef = 1 + 0.015 * (avg_temp - 25).clip(-5, 10)
    weekend_coef = np.where(is_weekend, 1.18, 1.0)
    uplift_coef  = 1 + 0.8 * discount_rate
    noise        = rng.normal(0, std * 0.5, n_days)

    sales_qty = (baseline * uplift_coef * weather_coef * weekend_coef + noise).clip(0).round().astype(int)

    return pd.DataFrame({
        "date":          dates.strftime("%Y-%m-%d"),
        "sku_id":        "SKU_COKE_330ML",
        "store_id":      store_id,
        "sales_qty":     sales_qty,
        "discount_rate": discount_rate.round(2),
        "is_promo":      is_promo.astype(int),
        "avg_temp":      avg_temp.round(1),
        "is_weekend":    is_weekend.astype(int),
        "baseline_qty":  baseline.round().astype(int),
    })


def generate(n_days: int = 540) -> pd.DataFrame:
    frames = [_generate_store(sid, n_days, RNG) for sid in _STORES]
    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    df = generate()
    df.to_csv(OUT, index=False)
    print(f"生成 {len(df)} 条记录 → {OUT}")
    for sid in _STORES:
        sub   = df[df["store_id"] == sid]
        promo = sub[sub["is_promo"] == 1]
        print(f"  {sid}: {len(sub)} 天  促销 uplift={promo['sales_qty'].mean()/promo['baseline_qty'].mean():.2f}x  "
              f"日均={sub['sales_qty'].mean():.0f}")
