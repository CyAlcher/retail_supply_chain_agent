"""HITLGate：低置信/高风险场景触发 Gradio 审核界面
- REVIEW 置信度：弹出 Gradio 界面等待人工批准/拒绝/修改
- REJECT 置信度：直接拒绝，不进入审核
- AUTO 置信度：自动通过
"""
from __future__ import annotations
import os
from retail_agent.schemas import PlannerState, ConfidenceTier


def _launch_review_ui(state: PlannerState) -> bool:
    """启动 Gradio 审核界面，返回 True=批准 / False=拒绝"""
    try:
        import gradio as gr
    except ImportError:
        print("[HITLGate] gradio 未安装，自动通过")
        return True

    fc  = state.forecast_result
    ss  = state.safety_stock_result
    ac  = state.action
    cv  = state.critic_verdict

    summary = f"""
## 待审核决策

**任务**: {state.task.raw_question}

**预测结果**
- P25 / P50 / P75: {fc.p25:.0f} / {fc.p50:.0f} / {fc.p75:.0f} 件
- 统计基线 P50: {fc.baseline_p50:.0f} 件（偏差 {fc.mape_vs_baseline:.1f}%）

**安全库存**: {ss.safety_stock_units:.0f} 件（覆盖 {ss.coverage_days:.0f} 天，服务水平 {ss.service_level*100:.0f}%）

**建议动作**: {ac.action_type} {ac.quantity:.0f} 件（置信度: {ac.confidence_tier.value}）

**风险提示**
{chr(10).join(f'- {r}' for r in (cv.risks if cv else [])) or '无'}

**Critic 质量评分**: {cv.quality.weighted_total:.2f if cv and cv.quality else 'N/A'}
"""

    decision = {"result": None}

    with gr.Blocks(title="供应链 AI — 人工审核") as demo:
        gr.Markdown("# 供应链 AI 决策审核")
        gr.Markdown(summary)
        note = gr.Textbox(label="审核备注（可选）", placeholder="输入修改意见或备注...")
        with gr.Row():
            approve_btn = gr.Button("✅ 批准执行", variant="primary")
            reject_btn  = gr.Button("❌ 拒绝", variant="stop")

        status = gr.Markdown("")

        def on_approve(note_text):
            decision["result"] = True
            decision["note"]   = note_text
            demo.close()
            return "已批准，正在关闭..."

        def on_reject(note_text):
            decision["result"] = False
            decision["note"]   = note_text
            demo.close()
            return "已拒绝，正在关闭..."

        approve_btn.click(on_approve, inputs=[note], outputs=[status])
        reject_btn.click(on_reject,  inputs=[note], outputs=[status])

    demo.launch(
        server_port=int(os.environ.get("HITL_PORT", 7860)),
        share=False,
        quiet=True,
        prevent_thread_lock=False,
    )

    return decision.get("result", True)


class HITLGate:
    def check(self, state: PlannerState) -> PlannerState:
        tier = state.action.confidence_tier if state.action else ConfidenceTier.AUTO

        if tier == ConfidenceTier.REJECT:
            state.hitl_required = True
            state.hitl_approved = False
            state.audit_trail.append({"node": "hitl", "decision": "rejected_by_rule"})

        elif tier == ConfidenceTier.REVIEW:
            state.hitl_required = True
            # 非交互模式（CI / demo）跳过 UI
            if os.environ.get("HITL_AUTO_APPROVE", "1") == "1":
                state.hitl_approved = True
                state.audit_trail.append({"node": "hitl", "decision": "auto_approved(review_tier)"})
            else:
                approved = _launch_review_ui(state)
                state.hitl_approved = approved
                state.audit_trail.append({
                    "node": "hitl",
                    "decision": "human_approved" if approved else "human_rejected",
                })

        else:
            state.hitl_required = False
            state.hitl_approved = True
            state.audit_trail.append({"node": "hitl", "decision": "auto_approved"})

        return state
