"""CriticAgent：规则评分（主路径）+ Claude LLM 风险识别与 reflection 决策
数值评分由规则确定性计算，LLM 只做语义层面的风险识别和反思决策。
这样评分稳定可控，LLM 的语义理解用在它擅长的地方。
"""
from __future__ import annotations
import os
import json
import anthropic
from retail_agent.schemas import (
    CriticVerdict, QualityScore, UncertaintyReport,
    ReflectionAction, ConfidenceTier, PlannerState,
)
from .base import BaseExpertAgent

_INTERVAL_THRESHOLDS = [(400, 1.0), (800, 0.8), (1200, 0.6), (2000, 0.4)]
_RETRY_MAX = 2

_RISK_SYSTEM = """你是零售供应链 AI 系统的风险识别专家。
根据预测上下文，识别业务层面的真实风险（不超过3条），并给出 reflection 决策。

## 风险识别规则
- 只报告真实的业务风险，不报告系统时序问题
- action 和 explain 在 Critic 之后生成，null 为正常时序，不是风险
- 关注：预测区间宽度、安全库存覆盖、气象预警、数据质量

## reflection 决策规则（严格遵守）
- forecast 存在 + 区间宽度 < 1200 + coverage_days >= 7 + 无高风险 → accept
- forecast 存在但区间宽度 1200~2000 或 coverage_days 3~7 → escalate_hitl
- forecast 为 null 或 P50 为负数 → abort
- 其他情况 → escalate_hitl

## 重要：大多数正常促销预测应该返回 accept，不要过度保守"""

_RISK_TOOL = {
    "name": "identify_risks",
    "description": "识别预测结果的业务风险并给出反思决策",
    "input_schema": {
        "type": "object",
        "properties": {
            "risks": {
                "type": "array",
                "items": {"type": "string"},
                "description": "业务风险列表（最多3条，只报告真实风险）",
            },
            "reflection": {
                "type": "string",
                "enum": ["accept", "retry_forecast", "escalate_hitl", "abort"],
            },
        },
        "required": ["risks", "reflection"],
    },
}


def _rule_score(state: PlannerState) -> QualityScore:
    """规则确定性评分，稳定可控"""
    fc = state.forecast_result
    ss = state.safety_stock_result

    # accuracy：区间宽度分级（更严格）
    if fc is None or (hasattr(fc, 'p50') and fc.p50 < 0):
        accuracy = 0.0
    else:
        interval = fc.p75 - fc.p25
        if interval < 400:
            accuracy = 1.0
        elif interval < 800:
            accuracy = 0.75
        elif interval < 1200:
            accuracy = 0.50
        elif interval < 2000:
            accuracy = 0.30
        else:
            accuracy = 0.10

    # completeness：必要字段齐全
    completeness = 1.0
    if not fc:
        completeness = 0.10
    elif not ss:
        completeness = 0.65

    # executability：visited_experts 覆盖率
    visited  = set(dict.fromkeys(
        a["expert"] for a in state.audit_trail if a.get("status") == "ok"
    ))
    required = set(state.plan_experts)
    coverage = len(visited & required) / max(len(required), 1)
    if coverage >= 0.8:
        executability = 1.0
    elif coverage >= 0.6:
        executability = 0.65
    elif coverage >= 0.4:
        executability = 0.45
    else:
        executability = 0.20

    # process_rationality：retry 次数
    if not visited:
        process = 0.05
    elif state.retry_count == 0:
        process = 1.0
    elif state.retry_count == 1:
        process = 0.75
    else:
        process = 0.50

    # business_value：安全库存覆盖天数（更严格）
    if ss is None:
        biz = 0.10
    elif ss.coverage_days >= 14:
        biz = 1.0
    elif ss.coverage_days >= 7:
        biz = 0.75
    elif ss.coverage_days >= 3:
        biz = 0.30   # 3~7天：明显不足，拉低总分
    else:
        biz = 0.05

    return QualityScore(
        accuracy=accuracy,
        completeness=completeness,
        compliance=1.0,
        executability=executability,
        process_rationality=round(process, 2),
        business_value=biz,
    )


def _llm_risks(state: PlannerState) -> tuple[list[str], ReflectionAction | None]:
    """调用 Claude 识别业务风险和 reflection 决策，失败返回 ([], None)"""
    try:
        fc = state.forecast_result
        ss = state.safety_stock_result
        wi = state.what_if_result

        context = {
            "forecast": {
                "p25": fc.p25, "p50": fc.p50, "p75": fc.p75,
                "interval_width": fc.p75 - fc.p25,
                "model": fc.model_used,
            } if fc else None,
            "safety_stock": {
                "units": ss.safety_stock_units,
                "coverage_days": ss.coverage_days,
                "service_level": ss.service_level,
            } if ss else None,
            "what_if_scenarios": len(wi.scenarios) if wi else 0,
            "timing_note": "action 和 explain 在 Critic 之后生成，null 为正常时序，不是风险",
            "plan_experts": state.plan_experts,
            "visited_experts": list(dict.fromkeys(
                a["expert"] for a in state.audit_trail if a.get("status") == "ok"
            )),
            "retry_count": state.retry_count,
        }

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            timeout=30.0,
            system=_RISK_SYSTEM,
            tools=[_RISK_TOOL],
            tool_choice={"type": "tool", "name": "identify_risks"},
            messages=[{
                "role": "user",
                "content": f"请识别以下预测结果的业务风险：\n{json.dumps(context, ensure_ascii=False)}",
            }],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "identify_risks":
                refl_map = {
                    "accept":         ReflectionAction.ACCEPT,
                    "retry_forecast": ReflectionAction.RETRY_FORECAST,
                    "escalate_hitl":  ReflectionAction.ESCALATE_HITL,
                    "abort":          ReflectionAction.ABORT,
                }
                risks = block.input.get("risks", [])
                reflection = refl_map.get(block.input.get("reflection", "accept"))
                return risks, reflection
    except Exception:
        pass
    return [], None


def _rule_risks(state: PlannerState) -> list[str]:
    """规则 fallback 风险识别"""
    fc = state.forecast_result
    ss = state.safety_stock_result
    risks = []
    if fc and (fc.p75 - fc.p25) > 1200:
        risks.append(f"预测区间过宽 ({fc.p75 - fc.p25:.0f})，不确定性高")
    if ss and ss.coverage_days < 7:
        risks.append(f"安全库存覆盖天数不足 ({ss.coverage_days:.1f} 天)")
    if state.task.weather and state.task.weather.alert_level:
        risks.append(f"气象预警: {state.task.weather.alert_level}")
    return risks


def _rule_reflection(quality: QualityScore, risks: list[str], retry_count: int,
                     state: "PlannerState") -> ReflectionAction:
    """规则 fallback reflection 决策"""
    fc = state.forecast_result
    score = quality.weighted_total

    # 区间过宽强制 escalate
    if fc and (fc.p75 - fc.p25) > 1200:
        return ReflectionAction.ESCALATE_HITL
    if score >= 0.80 and not risks:
        return ReflectionAction.ACCEPT
    if score < 0.60 and retry_count < _RETRY_MAX:
        return ReflectionAction.RETRY_FORECAST
    if risks:
        return ReflectionAction.ESCALATE_HITL
    return ReflectionAction.ACCEPT


class CriticAgent(BaseExpertAgent):
    name = "critic"

    def check_plan(self, state: PlannerState) -> tuple[bool, list[str]]:
        issues = []
        if not state.plan_experts:
            issues.append("plan_experts 为空，无法执行")
        if state.task.promo and "forecast" not in state.plan_experts:
            issues.append("促销场景必须包含 forecast expert")
        return len(issues) == 0, issues

    def score(self, state: PlannerState) -> tuple[QualityScore, list[str], UncertaintyReport]:
        fc = state.forecast_result

        # 规则评分（主路径，稳定可控）
        quality = _rule_score(state)

        # LLM 风险识别（语义层面）
        llm_risks, llm_reflection = _llm_risks(state)
        risks = llm_risks if llm_risks else _rule_risks(state)

        # 气象预警强制注入
        if state.task.weather and state.task.weather.alert_level:
            alert_msg = f"气象预警: {state.task.weather.alert_level}"
            if not any("气象" in r or "预警" in r for r in risks):
                risks.append(alert_msg)

        self._llm_reflection = llm_reflection

        interval = (fc.p75 - fc.p25) if fc else 0
        tier = ConfidenceTier.AUTO
        if risks:
            tier = ConfidenceTier.REVIEW
        if quality.weighted_total < 0.6:
            tier = ConfidenceTier.REJECT

        uncertainty = UncertaintyReport(
            p25=fc.p25 if fc else 0,
            p50=fc.p50 if fc else 0,
            p75=fc.p75 if fc else 0,
            interval_width=interval,
            confidence_tier=tier,
            alert_triggered=bool(risks),
            alert_reason="; ".join(risks),
        )
        return quality, risks, uncertainty

    def decide(self, quality: QualityScore, risks: list[str], retry_count: int,
               state: "PlannerState | None" = None) -> ReflectionAction:
        if self._llm_reflection is not None:
            if self._llm_reflection == ReflectionAction.RETRY_FORECAST and retry_count >= _RETRY_MAX:
                return ReflectionAction.ESCALATE_HITL if risks else ReflectionAction.ACCEPT
            return self._llm_reflection
        if state is not None:
            return _rule_reflection(quality, risks, retry_count, state)
        # fallback without state
        score = quality.weighted_total
        if score >= 0.80 and not risks:
            return ReflectionAction.ACCEPT
        if risks:
            return ReflectionAction.ESCALATE_HITL
        return ReflectionAction.ACCEPT

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        quality, risks, uncertainty = self.score(state)
        reflection = self.decide(quality, risks, state.retry_count, state)
        state.critic_verdict = CriticVerdict(
            quality=quality,
            uncertainty=uncertainty,
            risks=risks,
            reflection=reflection,
            retry_count=state.retry_count,
        )
        return state
