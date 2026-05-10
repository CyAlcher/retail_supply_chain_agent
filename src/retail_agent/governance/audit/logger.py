"""AuditLogger：结构化审计日志（演示版打印到终端）
TODO(真实化): 写入 Postgres audit 表 + Phoenix trace
"""
from __future__ import annotations
from retail_agent.schemas import PlannerState


class AuditLogger:
    def log(self, state: PlannerState) -> None:
        print(f"\n[Audit]  task_id={state.task.task_id}"
              f"  llm_calls={state.llm_call_count}(mock)"
              f"  tokens={state.total_tokens}"
              f"  hitl={'需要' if state.hitl_required else '自动通过'}"
              f"  error={state.error or 'none'}")
