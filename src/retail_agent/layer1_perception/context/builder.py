"""Layer1 上下文构建器：聚合历史数据 + 未来因子
TODO(真实化): 接入真实 Feature Store，替换合成数据
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from retail_agent.schemas import TaskContext

DATA_PATH = Path(__file__).parents[4] / "data" / "promo_synthetic" / "promo_history.csv"


def build(ctx: TaskContext) -> dict:
    """
    返回 {history_df, demand_std, avg_baseline, similar_promos}
    TODO(真实化): 从 Feature Store 按 sku_id + store_id 查询
    """
    if not DATA_PATH.exists():
        import subprocess, sys
        subprocess.run([sys.executable, str(DATA_PATH.parent / "gen_data.py")], check=True)

    df = pd.read_csv(DATA_PATH)
    df = df[(df["sku_id"] == ctx.sku_id) & (df["store_id"] == ctx.store_id)]

    demand_std   = float(df["sales_qty"].std())
    avg_baseline = float(df["baseline_qty"].mean())

    # 相似促销样本（折扣率 ±5%）
    if ctx.promo:
        dr = ctx.promo.discount_rate
        similar = df[
            (df["is_promo"] == 1) &
            (df["discount_rate"].between(dr - 0.05, dr + 0.05))
        ]
    else:
        similar = df[df["is_promo"] == 0]

    return {
        "history_df":    df,
        "demand_std":    demand_std,
        "avg_baseline":  avg_baseline,
        "similar_promos": similar,
        "n_similar":     len(similar),
    }
