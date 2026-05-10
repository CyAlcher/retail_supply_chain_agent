"""ActionBuilder：生成补货建议 + 置信度分级"""
from __future__ import annotations
from retail_agent.schemas import ActionRecommendation, ConfidenceTier, PlannerState


def build_action(state: PlannerState) -> ActionRecommendation:
    fc = state.forecast_result
    ss = state.safety_stock_result
    cv = state.critic_verdict

    p50 = fc.p50 if fc else 0
    safety = ss.safety_stock_units if ss else 0
    order_qty = round(p50 + safety)

    tier = ConfidenceTier.AUTO
    if cv and cv.uncertainty:
        tier = cv.uncertainty.confidence_tier

    rationale = f"预测中位数 {p50:.0f} 件 + 安全库存 {safety:.0f} 件"
    if cv and cv.risks:
        rationale += f"；风险提示: {cv.risks[0]}"

    return ActionRecommendation(
        action_type="下单",
        quantity=order_qty,
        confidence_tier=tier,
        rationale=rationale,
        forecast_p50=p50,
        safety_stock_units=safety,
    )
