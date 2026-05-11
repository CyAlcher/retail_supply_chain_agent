"""ExplainAgent：Claude 生成自然语言归因叙述
主路径用 Claude 生成差异化叙述，fallback 到模板拼接
"""
from __future__ import annotations
import os
import anthropic
from retail_agent.schemas import ExplainResult, AttributionFactor, PlannerState
from .base import BaseExpertAgent

_EXPLAIN_SYSTEM = (
    "你是零售供应链 AI 系统的归因解释专家。"
    "根据预测结果和驱动因子，用简洁的业务语言（2~3句话）解释本次预测的主要原因。"
    "语言要具体、有数字支撑，避免套话。直接输出叙述文字，不要加前缀。"
)


def _llm_narrative(factors: list[AttributionFactor], p50: float, task_info: dict) -> str | None:
    try:
        factor_text = "\n".join(
            f"- {f.factor}: {'+' if f.contribution_pct >= 0 else ''}{f.contribution_pct:.1f}% （{f.data_source}）"
            for f in factors
        )
        prompt = (
            f"预测中位数：{p50:.0f} 件\n"
            f"促销折扣：{task_info.get('discount_rate', 0)*100:.0f}%\n"
            f"气温：{task_info.get('avg_temp', 28)}℃\n"
            f"是否周末：{'是' if task_info.get('is_weekend') else '否'}\n\n"
            f"驱动因子分解：\n{factor_text}\n\n"
            "请生成归因叙述："
        )
        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=_EXPLAIN_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip() if resp.content else ""
        return text if text else None
    except Exception:
        return None


class ExplainAgent(BaseExpertAgent):
    name = "explain"

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        task = state.task
        fc   = state.forecast_result
        promo = task.promo

        factors: list[AttributionFactor] = []
        remaining = 100.0

        if promo and promo.discount_rate > 0:
            uplift_pct = round(promo.discount_rate * 0.8 * 100, 1)
            factors.append(AttributionFactor(
                factor="历史同力度促销 uplift",
                contribution_pct=uplift_pct,
                data_source="合成历史促销数据（相似折扣率样本）",
            ))
            remaining -= uplift_pct

        if task.weather and task.weather.avg_temp:
            temp_adj = round((task.weather.avg_temp - 25) * 0.015 * 100, 1)
            temp_adj = max(min(temp_adj, remaining * 0.4), -remaining * 0.4)
            factors.append(AttributionFactor(
                factor="天气调整因子",
                contribution_pct=round(temp_adj, 1),
                data_source=f"气温 {task.weather.avg_temp}℃",
            ))
            remaining -= abs(temp_adj)

        if task.calendar and task.calendar.is_weekend:
            weekend_pct = round(min(18.0, remaining * 0.5), 1)
            factors.append(AttributionFactor(
                factor="周末效应",
                contribution_pct=weekend_pct,
                data_source="历史周末 vs 工作日销量对比",
            ))
            remaining -= weekend_pct

        if remaining > 0:
            factors.append(AttributionFactor(
                factor="基线销量",
                contribution_pct=round(remaining, 1),
                data_source="近 90 天日均销量",
            ))

        task_info = {
            "discount_rate": promo.discount_rate if promo else 0,
            "avg_temp": task.weather.avg_temp if task.weather else 28,
            "is_weekend": task.calendar.is_weekend if task.calendar else False,
        }

        narrative = _llm_narrative(factors, fc.p50 if fc else 0, task_info)
        if not narrative:
            main = factors[0].factor if factors else "基线销量"
            narrative = (
                f"本次预测中位数 {fc.p50 if fc else 'N/A'} 件，"
                f"主要驱动因子为「{main}」。"
                f"建议关注促销陈列到位率，确保 uplift 兑现。"
            )

        state.explain_result = ExplainResult(
            key_drivers=factors,
            narrative=narrative,
            next_actions=["确认促销陈列到位", "提前 3 天下单备货", "监控首日销量偏差"],
        )
        return state
