"""Expert 基类：统一输入契约、审计钩子、降级接口"""
from __future__ import annotations
from abc import ABC, abstractmethod
from retail_agent.schemas import PlannerState


class BaseExpertAgent(ABC):
    name: str = "base"

    def run(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        state.audit_trail.append({"expert": self.name, "status": "start"})
        try:
            state = self._execute(state, ctx_data)
            state.audit_trail.append({"expert": self.name, "status": "ok"})
        except Exception as e:
            state.audit_trail.append({"expert": self.name, "status": "error", "msg": str(e)})
            state.error = f"{self.name}: {e}"
        return state

    @abstractmethod
    def _execute(self, state: PlannerState, ctx_data: dict) -> PlannerState:
        ...
