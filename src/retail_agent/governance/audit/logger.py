"""AuditLogger：结构化审计日志 + Arize Phoenix 可观测性
- 终端打印（始终）
- Phoenix trace（PHOENIX_ENABLED=1 时启用，自托管 http://localhost:6006）
"""
from __future__ import annotations
import os
import json
import time
from retail_agent.schemas import PlannerState

_tracer = None


def _get_tracer():
    global _tracer
    if _tracer is not None:
        return _tracer
    if os.environ.get("PHOENIX_ENABLED", "0") != "1":
        return None
    try:
        from phoenix.otel import register
        from opentelemetry import trace

        register(
            project_name="retail-supply-chain-agent",
            endpoint=os.environ.get("PHOENIX_ENDPOINT", "http://localhost:6006/v1/traces"),
        )
        _tracer = trace.get_tracer("retail_agent")
        return _tracer
    except Exception as e:
        print(f"[AuditLogger] Phoenix 初始化失败: {e}")
        return None


class AuditLogger:
    def log(self, state: PlannerState) -> None:
        # 终端打印
        print(f"\n[Audit]  task_id={state.task.task_id}"
              f"  llm_calls={state.llm_call_count}"
              f"  tokens={state.total_tokens}"
              f"  hitl={'需要' if state.hitl_required else '自动通过'}"
              f"  error={state.error or 'none'}")

        # Phoenix trace
        tracer = _get_tracer()
        if tracer is None:
            return

        try:
            fc = state.forecast_result
            ss = state.safety_stock_result
            cv = state.critic_verdict
            ac = state.action

            with tracer.start_as_current_span("retail_agent.decision") as span:
                span.set_attribute("task_id",      state.task.task_id)
                span.set_attribute("scenario",     state.task.scenario.value)
                span.set_attribute("sku_id",       state.task.sku_id)
                span.set_attribute("store_id",     state.task.store_id)
                span.set_attribute("llm_calls",    state.llm_call_count)
                span.set_attribute("total_tokens", state.total_tokens)
                span.set_attribute("error",        state.error or "")

                if fc:
                    span.set_attribute("forecast.p25",   fc.p25)
                    span.set_attribute("forecast.p50",   fc.p50)
                    span.set_attribute("forecast.p75",   fc.p75)
                    span.set_attribute("forecast.model", fc.model_used)
                    if fc.mape_vs_baseline is not None:
                        span.set_attribute("forecast.mape_vs_baseline", fc.mape_vs_baseline)

                if ss:
                    span.set_attribute("safety_stock.units",         ss.safety_stock_units)
                    span.set_attribute("safety_stock.coverage_days", ss.coverage_days)
                    span.set_attribute("safety_stock.service_level", ss.service_level)

                if cv and cv.quality:
                    span.set_attribute("critic.score",       cv.quality.weighted_total)
                    span.set_attribute("critic.accuracy",    cv.quality.accuracy)
                    span.set_attribute("critic.risks_count", len(cv.risks))
                    span.set_attribute("critic.reflection",  cv.reflection.value)

                if ac:
                    span.set_attribute("action.type",       ac.action_type)
                    span.set_attribute("action.quantity",   ac.quantity)
                    span.set_attribute("action.confidence", ac.confidence_tier.value)

                span.set_attribute("hitl.required", state.hitl_required)
                span.set_attribute("hitl.approved", state.hitl_approved or False)
                span.set_attribute("audit_trail",   json.dumps(state.audit_trail, ensure_ascii=False))

        except Exception as e:
            print(f"[AuditLogger] Phoenix trace 写入失败: {e}")
