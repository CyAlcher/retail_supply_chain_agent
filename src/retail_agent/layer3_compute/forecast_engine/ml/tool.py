"""LightGBM uplift 预测工具 + 统计基线对比
真实模型：在合成数据上训练，输出 P25/P50/P75 分位数预测。
统计基线：历史同期均值 + 促销系数，用于与 LightGBM 对比 MAPE。
TODO(真实化): 替换为从 Feature Store 读取真实 KA 客户历史数据
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

DATA_PATH = Path(__file__).parents[5] / "data" / "promo_synthetic" / "promo_history.csv"

_QUANTILES = [0.25, 0.50, 0.75]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["discount_rate", "avg_temp", "is_weekend", "is_promo"]].copy()
    X["temp_sq"] = X["avg_temp"] ** 2
    X["discount_x_temp"] = X["discount_rate"] * X["avg_temp"]
    return X


def _train_models(df: pd.DataFrame):
    try:
        import lightgbm as lgb
    except ImportError:
        return None, df["sales_qty"].std()

    X = _build_features(df)
    y = df["sales_qty"].values
    models = {}
    for q in _QUANTILES:
        m = lgb.LGBMRegressor(
            objective="quantile",
            alpha=q,
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=15,
            verbose=-1,
        )
        m.fit(X, y)
        models[q] = m
    return models, df["sales_qty"].std()


def _baseline_predict(
    df: pd.DataFrame,
    discount_rate: float,
    avg_temp: float,
    is_weekend: bool,
    duration_days: int,
) -> dict:
    """统计基线：历史同期均值 × 促销系数 × 天气系数 × 周末系数"""
    baseline_daily = float(df["baseline_qty"].mean())
    uplift = 1 + 0.8 * discount_rate
    weather_adj = 1 + 0.015 * (avg_temp - 25)
    weekend_adj = 1.18 if is_weekend else 1.0
    p50 = baseline_daily * uplift * weather_adj * weekend_adj * duration_days
    std = float(df["sales_qty"].std()) * (duration_days ** 0.5)
    return {
        "p25": round(max(p50 - 0.674 * std, 0), 1),
        "p50": round(max(p50, 0), 1),
        "p75": round(p50 + 0.674 * std, 1),
        "model_used": "statistical_baseline",
    }


_MODELS = None
_DEMAND_STD: float = 80.0
_HISTORY_DF: pd.DataFrame | None = None


def _ensure_loaded() -> None:
    global _MODELS, _DEMAND_STD, _HISTORY_DF
    if _MODELS is not None:
        return
    if not DATA_PATH.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_PATH.parent / "gen_data.py")], check=True)
    _HISTORY_DF = pd.read_csv(DATA_PATH)
    _MODELS, _DEMAND_STD = _train_models(_HISTORY_DF)


def predict(
    discount_rate: float,
    avg_temp: float,
    is_weekend: bool,
    duration_days: int = 3,
    model_family: str = "lgb_quantile_uplift",
) -> dict:
    """
    返回 {p25, p50, p75, model_used, demand_std, feature_importance, baseline_p50, mape_vs_baseline}
    model_family: "lgb_quantile_uplift" | "statistical_baseline"
    """
    _ensure_loaded()

    # 统计基线（始终计算，用于对比）
    baseline = _baseline_predict(_HISTORY_DF, discount_rate, avg_temp, is_weekend, duration_days)

    if model_family == "statistical_baseline" or not _MODELS:
        return {
            **baseline,
            "demand_std":         round(_DEMAND_STD * (duration_days ** 0.5), 1),
            "feature_importance": {},
            "baseline_p50":       baseline["p50"],
            "mape_vs_baseline":   0.0,
        }

    row = pd.DataFrame([{
        "discount_rate": discount_rate,
        "avg_temp":      avg_temp,
        "is_weekend":    int(is_weekend),
        "is_promo":      1,
    }])
    X = _build_features(row)

    p25 = float(_MODELS[0.25].predict(X)[0]) * duration_days
    p50 = float(_MODELS[0.50].predict(X)[0]) * duration_days
    p75 = float(_MODELS[0.75].predict(X)[0]) * duration_days

    fi = dict(zip(X.columns, _MODELS[0.50].feature_importances_))
    fi_norm = {k: round(v / max(sum(fi.values()), 1), 3) for k, v in fi.items()}

    mape = abs(p50 - baseline["p50"]) / max(baseline["p50"], 1) * 100

    return {
        "p25":                round(max(p25, 0), 1),
        "p50":                round(max(p50, 0), 1),
        "p75":                round(max(p75, 0), 1),
        "model_used":         "lgb_quantile_uplift",
        "demand_std":         round(_DEMAND_STD * (duration_days ** 0.5), 1),
        "feature_importance": fi_norm,
        "baseline_p50":       baseline["p50"],
        "mape_vs_baseline":   round(mape, 1),
    }

