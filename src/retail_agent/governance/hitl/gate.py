"""HITLGate：演示版自动通过
TODO(真实化): 接入真实审核界面（Web/IM），等待人工确认
"""
from __future__ import annotations
from retail_agent.schemas import PlannerState, ConfidenceTier


class HITLGate:
    def check(self, state: PlannerState) -> PlannerState:
        # TODO(真实化): 低置信/高金额时暂停等待人工审核
        if state.action and state.action.confidence_tier == ConfidenceTier.REJECT:
            state.hitl_required = True
            state.hitl_approved = False
            state.audit_trail.append({"node": "hitl", "decision": "rejected_by_rule"})
        else:
            state.hitl_required = False
            state.hitl_approved = True
            state.audit_trail.append({"node": "hitl", "decision": "auto_approved"})
        return state
