"""CriticAgent：规则驱动的质量评估（6 项职责）
TODO(真实化): 用 Claude LLM-as-Judge + 结构化 Rubric 替换规则判断
"""
from __future__ import annotations
from retail_agent.schemas import (
    CriticVerdict, QualityScore, UncertaintyReport,
    ReflectionAction, ConfidenceTier, PlannerState,
)
from .base import BaseExpertAgent

_WAPE_THRESHOLD   = 0.30   # 超过则降档
_INTERVAL_MAX     = 1200   # P75-P25 超过此值视为不确定性过高
_RETRY_MAX        = 2


class CriticAgent(BaseExpertAgent):
    name = "critic"

    # ── 职责① 计划合理性校验 ──────────────────────────────────────────────────
    def check_plan(self, state: PlannerState) -> tuple[bool, list[str]]:
        issues = []
        if not state.plan_experts:
            issues.append("plan_experts 为空，无法执行")
        if state.task.promo and "forecast" not in state.plan_experts:
            issues.append("促销场景必须包含 forecast expert")
        return len(issues) == 0, issues

    # ── 职责②③④ 质量评分 + 风险识别 + 不确定性量化 ──────────────────────────
    def score(self, state: PlannerState) -> tuple[QualityScore, list[str], UncertaintyReport]:
        fc = state.forecast_result
        ss = state.safety_stock_result

        # 准确性：区间宽度合理性代理（无真值时）
        interval = (fc.p75 - fc.p25) if fc else 0
        accuracy = 1.0 if interval < 600 else (0.7 if interval < _INTERVAL_MAX else 0.4)

        # 完备性：必要字段是否齐全
        completeness = 1.0
        if not fc:
            completeness -= 0.5
        if not ss:
            completeness -= 0.3

        # 合规性：禁用工具未被调用（mock 下默认合规）
        compliance = 1.0

        # 可执行性：有建议动作
        executability = 1.0 if state.action or state.explain_result else 0.6

        # 过程合理性：决策链节点覆盖
        visited = {a["expert"] for a in state.audit_trail if a.get("status") == "ok"}
        required = set(state.plan_experts)
        process = len(visited & required) / max(len(required), 1)

        # 业务价值：安全库存覆盖天数 ≥ 7
        biz = 1.0 if (ss and ss.coverage_days >= 7) else 0.6

        quality = QualityScore(
            accuracy=accuracy,
            completeness=completeness,
            compliance=compliance,
            executability=executability,
            process_rationality=round(process, 2),
            business_value=biz,
        )

        # 风险识别
        risks = []
        if fc and (fc.p75 - fc.p25) > _INTERVAL_MAX:
            risks.append(f"预测区间过宽 ({fc.p75 - fc.p25:.0f})，不确定性高")
        if ss and ss.coverage_days < 7:
            risks.append(f"安全库存覆盖天数不足 ({ss.coverage_days:.1f} 天)")
        if state.task.weather and state.task.weather.alert_level:
            risks.append(f"气象预警: {state.task.weather.alert_level}")

        # 不确定性量化
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

    # ── 职责⑤ 反思决策 ───────────────────────────────────────────────────────
    def decide(self, quality: QualityScore, risks: list[str], retry_count: int) -> ReflectionAction:
        # TODO(真实化): 用 Claude 做结构化决策
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
