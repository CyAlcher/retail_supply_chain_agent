"""LightGBM uplift 预测工具
真实模型：在合成数据上训练，输出 P25/P50/P75 分位数预测。
TODO(真实化): 替换为从 Feature Store 读取真实 KA 客户历史数据
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import norm

DATA_PATH = Path(__file__).parents[5] / "data" / "promo_synthetic" / "promo_history.csv"

# 分位数回归用三个 alpha
_QUANTILES = [0.25, 0.50, 0.75]


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[["discount_rate", "avg_temp", "is_weekend", "is_promo"]].copy()
    X["temp_sq"] = X["avg_temp"] ** 2
    X["discount_x_temp"] = X["discount_rate"] * X["avg_temp"]
    return X


def _train_models(df: pd.DataFrame):
    """训练三个分位数 LightGBM 模型，返回 (models, baseline_std)"""
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


# 模块级缓存，避免每次调用重新训练
_MODELS = None
_DEMAND_STD: float = 80.0
_HISTORY_DF: pd.DataFrame | None = None


def _ensure_loaded() -> None:
    global _MODELS, _DEMAND_STD, _HISTORY_DF
    if _MODELS is not None:
        return
    if not DATA_PATH.exists():
        # 数据不存在时先生成
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_PATH.parent / "gen_data.py")], check=True)
    _HISTORY_DF = pd.read_csv(DATA_PATH)
    _MODELS, _DEMAND_STD = _train_models(_HISTORY_DF)


def predict(
    discount_rate: float,
    avg_temp: float,
    is_weekend: bool,
    duration_days: int = 3,
) -> dict:
    """
    返回 {p25, p50, p75, model_used, demand_std, feature_importance}
    TODO(真实化): 接入真实 Feature Store，替换合成数据
    """
    _ensure_loaded()

    row = pd.DataFrame([{
        "discount_rate": discount_rate,
        "avg_temp":      avg_temp,
        "is_weekend":    int(is_weekend),
        "is_promo":      1,
    }])
    X = _build_features(row)

    if _MODELS:
        p25 = float(_MODELS[0.25].predict(X)[0]) * duration_days
        p50 = float(_MODELS[0.50].predict(X)[0]) * duration_days
        p75 = float(_MODELS[0.75].predict(X)[0]) * duration_days
        model_used = "lgb_quantile_uplift"
        # 简单特征重要性（P50 模型）
        fi = dict(zip(X.columns, _MODELS[0.50].feature_importances_))
        fi_norm = {k: round(v / max(sum(fi.values()), 1), 3) for k, v in fi.items()}
    else:
        # fallback: 统计估算
        base = 600 * (1 + 0.8 * discount_rate)
        weather_adj = 1 + 0.015 * (avg_temp - 25)
        weekend_adj = 1.18 if is_weekend else 1.0
        p50 = base * weather_adj * weekend_adj * duration_days
        p25 = p50 * 0.85
        p75 = p50 * 1.15
        model_used = "statistical_fallback"
        fi_norm = {}

    return {
        "p25":                round(max(p25, 0), 1),
        "p50":                round(max(p50, 0), 1),
        "p75":                round(max(p75, 0), 1),
        "model_used":         model_used,
        "demand_std":         round(_DEMAND_STD * (duration_days ** 0.5), 1),
        "feature_importance": fi_norm,
    }
