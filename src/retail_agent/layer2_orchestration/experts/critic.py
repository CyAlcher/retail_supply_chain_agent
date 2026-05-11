"""CriticAgent：LLM-as-Judge 质量评估（6 项职责）
主路径用 Claude structured output 评分，fallback 到规则判断
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

_WAPE_THRESHOLD = 0.30
_INTERVAL_MAX   = 1200
_RETRY_MAX      = 2

_RUBRIC_SYSTEM = """你是零售供应链 AI 系统的质量评审官。
对预测结果按六个维度打分（0.0~1.0），返回 JSON。

评分维度：
- accuracy: 预测区间合理性（区间越窄、P50 越合理分越高）
- completeness: 必要字段齐全程度
- compliance: 无违规工具调用
- executability: 输出是否可直接执行
- process_rationality: 决策链节点覆盖完整度
- business_value: 安全库存覆盖天数是否满足业务需求

同时输出：
- risks: 风险列表（字符串数组）
- reflection: "accept" | "retry_forecast" | "escalate_hitl" | "abort"

严格返回 JSON，不要其他文字。"""

_RUBRIC_TOOL = {
    "name": "quality_score",
    "description": "对预测结果进行六维质量评分",
    "input_schema": {
        "type": "object",
        "properties": {
            "accuracy":            {"type": "number", "minimum": 0, "maximum": 1},
            "completeness":        {"type": "number", "minimum": 0, "maximum": 1},
            "compliance":          {"type": "number", "minimum": 0, "maximum": 1},
            "executability":       {"type": "number", "minimum": 0, "maximum": 1},
            "process_rationality": {"type": "number", "minimum": 0, "maximum": 1},
            "business_value":      {"type": "number", "minimum": 0, "maximum": 1},
            "risks":               {"type": "array", "items": {"type": "string"}},
            "reflection":          {"type": "string",
                                    "enum": ["accept", "retry_forecast", "escalate_hitl", "abort"]},
        },
        "required": ["accuracy", "completeness", "compliance", "executability",
                     "process_rationality", "business_value", "risks", "reflection"],
    },
}


def _llm_score(state: PlannerState) -> dict | None:
    """调用 Claude LLM-as-Judge，返回评分 dict，失败返回 None"""
    try:
        fc = state.forecast_result
        ss = state.safety_stock_result
        wi = state.what_if_result
        ac = state.action

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
            "action": ac.action_type if ac else None,
            "plan_experts": state.plan_experts,
            "visited_experts": [a["expert"] for a in state.audit_trail if a.get("status") == "ok"],
            "retry_count": state.retry_count,
        }

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=_RUBRIC_SYSTEM,
            tools=[_RUBRIC_TOOL],
            tool_choice={"type": "tool", "name": "quality_score"},
            messages=[{
                "role": "user",
                "content": f"请对以下预测结果进行质量评分：\n{json.dumps(context, ensure_ascii=False)}",
            }],
        )
        for block in resp.content:
            if block.type == "tool_use" and block.name == "quality_score":
                return block.input
    except Exception:
        pass
    return None


def _rule_score(state: PlannerState) -> tuple[QualityScore, list[str]]:
    """规则 fallback 评分"""
    fc = state.forecast_result
    ss = state.safety_stock_result

    interval = (fc.p75 - fc.p25) if fc else 0
    accuracy = 1.0 if interval < 600 else (0.7 if interval < _INTERVAL_MAX else 0.4)

    completeness = 1.0
    if not fc:
        completeness -= 0.5
    if not ss:
        completeness -= 0.3

    executability = 1.0 if state.action or state.explain_result else 0.6

    visited  = {a["expert"] for a in state.audit_trail if a.get("status") == "ok"}
    required = set(state.plan_experts)
    process  = len(visited & required) / max(len(required), 1)

    biz = 1.0 if (ss and ss.coverage_days >= 7) else 0.6

    quality = QualityScore(
        accuracy=accuracy, completeness=completeness, compliance=1.0,
        executability=executability, process_rationality=round(process, 2),
        business_value=biz,
    )

    risks = []
    if fc and (fc.p75 - fc.p25) > _INTERVAL_MAX:
        risks.append(f"预测区间过宽 ({fc.p75 - fc.p25:.0f})，不确定性高")
    if ss and ss.coverage_days < 7:
        risks.append(f"安全库存覆盖天数不足 ({ss.coverage_days:.1f} 天)")
    if state.task.weather and state.task.weather.alert_level:
        risks.append(f"气象预警: {state.task.weather.alert_level}")

    return quality, risks


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

        llm_result = _llm_score(state)
        if llm_result:
            reflection_map = {
                "accept":         ReflectionAction.ACCEPT,
                "retry_forecast": ReflectionAction.RETRY_FORECAST,
                "escalate_hitl":  ReflectionAction.ESCALATE_HITL,
                "abort":          ReflectionAction.ABORT,
            }
            quality = QualityScore(
                accuracy=llm_result["accuracy"],
                completeness=llm_result["completeness"],
                compliance=llm_result["compliance"],
                executability=llm_result["executability"],
                process_rationality=llm_result["process_rationality"],
                business_value=llm_result["business_value"],
            )
            risks = llm_result.get("risks", [])
            self._llm_reflection = reflection_map.get(
                llm_result.get("reflection", "accept"), ReflectionAction.ACCEPT
            )
        else:
            quality, risks = _rule_score(state)
            self._llm_reflection = None

        # 气象预警强制注入，不依赖 LLM 判断
        if state.task.weather and state.task.weather.alert_level:
            alert_msg = f"气象预警: {state.task.weather.alert_level}"
            if not any("气象" in r or "预警" in r for r in risks):
                risks.append(alert_msg)

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

    def decide(self, quality: QualityScore, risks: list[str], retry_count: int) -> ReflectionAction:
        # 优先使用 LLM 的反思决策
        if hasattr(self, "_llm_reflection") and self._llm_reflection is not None:
            # 但 retry 次数超限时强制 accept 或 escalate
            if self._llm_reflection == ReflectionAction.RETRY_FORECAST and retry_count >= _RETRY_MAX:
                return ReflectionAction.ESCALATE_HITL if risks else ReflectionAction.ACCEPT
            return self._llm_reflection

        # fallback 规则
        if quality.weighted_total >= 0.75 and not any("过宽" in r for r in risks):
            return ReflectionAction.ACCEPT
        if retry_count < _RETRY_MAX and quality.accuracy < 0.6:
            return ReflectionAction.RETRY_FORECAST
        if risks:
            return ReflectionAction.ESCALATE_HITL
        return ReflectionAction.ACCEPT

    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        quality, risks, uncertainty = self.score(state)
        reflection = self.decide(quality, risks, state.retry_count)
        state.critic_verdict = CriticVerdict(
            quality=quality,
            uncertainty=uncertainty,
            risks=risks,
            reflection=reflection,
            retry_count=state.retry_count,
        )
        return state
