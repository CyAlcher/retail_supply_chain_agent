"""LangGraph StateGraph 主编排图
真实 LangGraph 图结构，SqliteSaver checkpointer（支持决策回放）
"""
from __future__ import annotations
import os
import sqlite3
from typing import Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from retail_agent.schemas import PlannerState, Scenario, ReflectionAction
from retail_agent.layer1_perception.context import builder as ctx_builder
from retail_agent.layer2_orchestration.experts.forecast import ForecastAgent
from retail_agent.layer2_orchestration.experts.safety_stock import SafetyStockAgent
from retail_agent.layer2_orchestration.experts.what_if import WhatIfAgent
from retail_agent.layer2_orchestration.experts.explain import ExplainAgent
from retail_agent.layer2_orchestration.experts.critic import CriticAgent
from retail_agent.layer4_decision.action_builder.builder import build_action
from retail_agent.governance.hitl.gate import HITLGate
from retail_agent.governance.audit.logger import AuditLogger

_forecast_agent    = ForecastAgent()
_safety_agent      = SafetyStockAgent()
_whatif_agent      = WhatIfAgent()
_explain_agent     = ExplainAgent()
_critic_agent      = CriticAgent()
_hitl_gate         = HITLGate()
_audit_logger      = AuditLogger()


def _state_to_dict(s: PlannerState) -> dict[str, Any]:
    return s.model_dump()

def _dict_to_state(d: dict[str, Any]) -> PlannerState:
    return PlannerState.model_validate(d)


# ── 节点函数（LangGraph 要求返回 dict） ───────────────────────────────────────

def node_plan(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    experts = ["forecast", "safety_stock", "explain"]
    if s.task.scenario == Scenario.PROMO:
        experts.append("what_if")
    s.plan_experts = experts
    s.audit_trail.append({"node": "plan", "experts": experts})
    return _state_to_dict(s)


def node_critic_check_plan(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    ok, issues = _critic_agent.check_plan(s)
    s.audit_trail.append({"node": "critic_check_plan", "ok": ok, "issues": issues})
    if not ok:
        s.error = f"计划校验失败: {issues}"
    return _state_to_dict(s)


def node_forecast(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    s = _forecast_agent.run(s, ctx_data)
    return _state_to_dict(s)


def node_safety_stock(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    s = _safety_agent.run(s, ctx_data)
    return _state_to_dict(s)


def node_what_if(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    s = _whatif_agent.run(s, ctx_data)
    return _state_to_dict(s)


def node_critic_score(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    s = _critic_agent.run(s, ctx_data)
    return _state_to_dict(s)


def node_explain(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    ctx_data = ctx_builder.build(s.task)
    s = _explain_agent.run(s, ctx_data)
    return _state_to_dict(s)


def node_action_builder(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    s.action = build_action(s)
    s.audit_trail.append({"node": "action_builder", "action": s.action.action_type if s.action else None})
    return _state_to_dict(s)


def node_hitl(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    s = _hitl_gate.check(s)
    return _state_to_dict(s)


def node_audit(state_dict: dict) -> dict:
    s = _dict_to_state(state_dict)
    _audit_logger.log(s)
    return _state_to_dict(s)


# ── 条件路由 ──────────────────────────────────────────────────────────────────

def route_after_plan(state_dict: dict) -> str:
    s = _dict_to_state(state_dict)
    if s.error:
        return "audit"
    return "critic_check_plan"


def route_after_critic_check(state_dict: dict) -> str:
    s = _dict_to_state(state_dict)
    if s.error:
        return "audit"
    return "forecast"


def route_after_critic_score(state_dict: dict) -> str:
    s = _dict_to_state(state_dict)
    v = s.critic_verdict
    if not v:
        return "explain"
    action = v.reflection
    if action == ReflectionAction.RETRY_FORECAST and s.retry_count < 2:
        s.retry_count += 1
        return "forecast"
    if action == ReflectionAction.ESCALATE_HITL:
        return "hitl"
    if action == ReflectionAction.ABORT:
        return "audit"
    return "explain"


def route_what_if(state_dict: dict) -> str:
    """促销场景才跑 what_if，否则直接 critic_score"""
    s = _dict_to_state(state_dict)
    if s.task.scenario == Scenario.PROMO and "what_if" in s.plan_experts:
        return "what_if"
    return "critic_score"


# ── 构建图 ────────────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(dict)

    g.add_node("plan",              node_plan)
    g.add_node("critic_check_plan", node_critic_check_plan)
    g.add_node("forecast",          node_forecast)
    g.add_node("safety_stock",      node_safety_stock)
    g.add_node("what_if",           node_what_if)
    g.add_node("critic_score",      node_critic_score)
    g.add_node("explain",           node_explain)
    g.add_node("action_builder",    node_action_builder)
    g.add_node("hitl",              node_hitl)
    g.add_node("audit",             node_audit)

    g.set_entry_point("plan")
    g.add_conditional_edges("plan", route_after_plan,
                             {"critic_check_plan": "critic_check_plan", "audit": "audit"})
    g.add_conditional_edges("critic_check_plan", route_after_critic_check,
                             {"forecast": "forecast", "audit": "audit"})
    g.add_edge("forecast", "safety_stock")
    g.add_conditional_edges("safety_stock", route_what_if,
                             {"what_if": "what_if", "critic_score": "critic_score"})
    g.add_edge("what_if", "critic_score")
    g.add_conditional_edges("critic_score", route_after_critic_score, {
        "forecast": "forecast",
        "hitl":     "hitl",
        "audit":    "audit",
        "explain":  "explain",
    })
    g.add_edge("explain",        "action_builder")
    g.add_edge("action_builder", "hitl")
    g.add_edge("hitl",           "audit")
    g.add_edge("audit",          END)

    db_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "..", "..", "data", "checkpoints.db"
    ))
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    checkpointer.setup()
    return g.compile(checkpointer=checkpointer)


GRAPH = build_graph()


def run(state: PlannerState) -> PlannerState:
    config = {"configurable": {"thread_id": state.task.task_id}}
    result = GRAPH.invoke(_state_to_dict(state), config=config)
    return _dict_to_state(result)


def replay(task_id: str) -> PlannerState | None:
    """重放历史决策，从 SQLite checkpoint 恢复最终状态"""
    config = {"configurable": {"thread_id": task_id}}
    result = GRAPH.get_state(config)
    if result and result.values:
        return _dict_to_state(result.values)
    return None
