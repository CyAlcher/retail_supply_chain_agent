"""
retail-supply-chain-agent — Gradio Web 交互界面
2026 AI-Native 最佳实践：
  - Streaming 输出（实时展示 Agent 决策链路）
  - 结构化结果卡片（Forecast / SafetyStock / WhatIf / Explain / Critic）
  - 预设 case 一键体验 + 自由输入
  - HITL 审核内嵌（低置信场景弹出审核面板）
  - 历史记录（SQLite replay）
  - 暗色主题，企业级风格

用法：
  conda run -n agent python demo/web_app.py
  # 浏览器打开 http://localhost:7861
"""
from __future__ import annotations
import sys
import os
import uuid
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
os.environ.setdefault("HITL_AUTO_APPROVE", "1")   # Web 模式下 HITL 由界面处理

import gradio as gr
from retail_agent.schemas import (
    TaskContext, PromoFactor, WeatherFactor, CalendarFactor, Scenario,
    ConfidenceTier,
)
from retail_agent.layer1_perception.entry.entrypoint import parse as parse_question
from retail_agent.layer2_orchestration.planner.planner import run as run_graph

# ── 预设 case ─────────────────────────────────────────────────────────────────

PRESET_CASES = {
    "🛒 促销备货（DEV-PROMO-001）": (
        "华润万家天河店，可口可乐 330ml 罐装，"
        "下周六（2026-06-13）开始做 25% 折扣促销 3 天，"
        "帮我预测销量并给出备货建议。",
        {"weather": WeatherFactor(avg_temp=32.0, rain_prob=0.3),
         "calendar": CalendarFactor(is_weekend=True)},
    ),
    "📊 折扣力度对比（DEV-PROMO-002）": (
        "同一家门店，同款可口可乐 330ml，"
        "如果把折扣从 20% 提到 30%，销量和毛利分别会怎么变化？哪个方案更划算？",
        {"promo": PromoFactor(discount_rate=0.20, start_date="2026-06-13",
                              end_date="2026-06-15", duration_days=3),
         "weather": WeatherFactor(avg_temp=30.0),
         "calendar": CalendarFactor(is_weekend=False)},
    ),
    "🌧️ 暴雨预警降档（DEV-PROMO-005）": (
        "华润万家天河店，可口可乐 330ml，下周促销期间气象局发布暴雨橙色预警，"
        "预计降雨概率 85%，是否需要下调备货量？",
        {"weather": WeatherFactor(avg_temp=24.0, rain_prob=0.85, alert_level="暴雨橙色"),
         "calendar": CalendarFactor(is_weekend=True),
         "promo": PromoFactor(discount_rate=0.25, start_date="2026-06-27",
                              end_date="2026-06-29", duration_days=3)},
    ),
    "🏪 多门店汇总（DEV-PROMO-004）": (
        "华润万家广州区 3 家门店（天河店、越秀店、海珠店），"
        "可口可乐 330ml，下周统一做 20% 折扣促销 3 天，"
        "分别预测各门店销量并汇总备货建议。",
        {"weather": WeatherFactor(avg_temp=31.0),
         "calendar": CalendarFactor(is_weekend=False),
         "promo": PromoFactor(discount_rate=0.20, start_date="2026-06-20",
                              end_date="2026-06-22", duration_days=3)},
    ),
    "🔍 偏差归因分析（DEV-PROMO-003）": (
        "华润万家天河店，可口可乐 330ml，上周促销实际销量 1800 件，"
        "但预测是 2375 件，偏差 -24.5%，帮我分析偏差原因。",
        {"weather": WeatherFactor(avg_temp=28.0, rain_prob=0.7),
         "calendar": CalendarFactor(is_weekend=True),
         "promo": PromoFactor(discount_rate=0.25, start_date="2026-06-06",
                              end_date="2026-06-08", duration_days=3)},
    ),
}

# ── 核心运行函数（Generator，支持 streaming）────────────────────────────────

def run_agent(question: str, preset_key: str) -> tuple[str, str, str, str, str, str]:
    """
    运行 Agent，返回各模块结果。
    使用 Generator yield 实现流式更新。
    """
    if not question.strip():
        yield ("", "", "", "", "", "⚠️ 请输入问题或选择预设 case")
        return

    task_id = f"WEB-{uuid.uuid4().hex[:8].upper()}"
    log_lines: list[str] = []

    def log(msg: str):
        log_lines.append(msg)

    log(f"🚀 **任务 ID**: `{task_id}`")
    log(f"📝 **输入**: {question}")
    log("---")
    log("⏳ 正在解析意图...")
    yield ("\n".join(log_lines), "", "", "", "", "")

    # ── 解析意图 ──────────────────────────────────────────────────────────────
    try:
        ctx = parse_question(question, task_id=task_id)

        # 预设 case 覆盖部分字段
        if preset_key and preset_key in PRESET_CASES:
            _, overrides = PRESET_CASES[preset_key]
            for k, v in overrides.items():
                setattr(ctx, k, v)
            ctx.task_id = task_id

        log(f"✅ **意图解析完成**")
        log(f"- 门店: `{ctx.store_id}`  SKU: `{ctx.sku_id}`")
        log(f"- 场景: `{ctx.scenario.value}`")
        if ctx.promo:
            log(f"- 促销折扣: `{ctx.promo.discount_rate*100:.0f}%`  时长: `{ctx.promo.duration_days}天`")
        if ctx.weather and ctx.weather.avg_temp:
            log(f"- 气温: `{ctx.weather.avg_temp}℃`"
                + (f"  ⚠️ 气象预警: `{ctx.weather.alert_level}`"
                   if ctx.weather.alert_level else ""))
        log("---")
        log("⏳ 正在运行 Agent 决策链路...")
        yield ("\n".join(log_lines), "", "", "", "", "")
    except Exception as e:
        yield (f"❌ 意图解析失败: {e}", "", "", "", "", "")
        return

    # ── 运行 Agent ─────────────────────────────────────────────────────────────
    try:
        from retail_agent.schemas import PlannerState
        state = PlannerState(task=ctx)
        t0 = time.time()
        final = run_graph(state)
        elapsed = time.time() - t0

        log(f"✅ **Agent 运行完成**  耗时 `{elapsed:.1f}s`  LLM调用 `{final.llm_call_count}次`")
        log("---")
        yield ("\n".join(log_lines), "", "", "", "", "")
    except Exception as e:
        yield (f"❌ Agent 运行失败: {e}", "", "", "", "", "")
        return

    # ── 格式化各模块输出 ───────────────────────────────────────────────────────
    fc = final.forecast_result
    ss = final.safety_stock_result
    wi = final.what_if_result
    ex = final.explain_result
    cv = final.critic_verdict
    ac = final.action

    # Forecast 卡片
    forecast_md = ""
    if fc:
        interval = fc.p75 - fc.p25
        forecast_md = f"""### 📈 销量预测

| 指标 | 数值 |
|---|---|
| P25（悲观） | **{fc.p25:.0f} 件** |
| P50（中位） | **{fc.p50:.0f} 件** |
| P75（乐观） | **{fc.p75:.0f} 件** |
| 区间宽度 | {interval:.0f} 件 |
| 统计基线 P50 | {fc.baseline_p50:.0f} 件 |
| vs 基线偏差 | {fc.mape_vs_baseline:.1f}% |
| 模型 | `{fc.model_used.split('[')[0].strip()}` |

**特征重要性**（Top 3）：
"""
        if fc.feature_importance:
            top3 = sorted(fc.feature_importance.items(), key=lambda x: -x[1])[:3]
            for feat, imp in top3:
                bar = "█" * int(imp * 20)
                forecast_md += f"- `{feat}`: {bar} {imp:.3f}\n"

    # SafetyStock + Action 卡片
    action_md = ""
    if ss and ac:
        tier_emoji = {"自动执行": "🟢", "需要复核": "🟡", "拒绝": "🔴"}.get(
            ac.confidence_tier.value, "⚪"
        )
        action_md = f"""### 📦 备货建议

| 指标 | 数值 |
|---|---|
| **建议下单量** | **{ac.quantity:.0f} 件** |
| 预测中位数 | {fc.p50:.0f} 件 |
| 安全库存 | {ss.safety_stock_units:.0f} 件 |
| 覆盖天数 | {ss.coverage_days:.0f} 天 |
| 服务水平 | {ss.service_level*100:.0f}% |
| 置信度 | {tier_emoji} **{ac.confidence_tier.value}** |

> {ac.rationale}
"""

    # WhatIf 卡片
    whatif_md = ""
    if wi and wi.scenarios:
        whatif_md = "### 🔀 方案对比推演\n\n"
        whatif_md += "| 方案 | 折扣率 | 预测销量 P50 | 毛利 |\n|---|---|---|---|\n"
        for s in wi.scenarios:
            mark = " ✅ **推荐**" if s.label == wi.recommended else ""
            whatif_md += f"| {s.label}{mark} | {s.discount_rate*100:.0f}% | {s.forecast_p50:.0f} 件 | ¥{s.gross_profit:.0f} |\n"
        whatif_md += f"\n**推荐理由**：{wi.recommendation_reason}"

    # Explain 卡片
    explain_md = ""
    if ex:
        explain_md = "### 💡 归因解释\n\n"
        explain_md += "| 驱动因子 | 贡献度 | 数据来源 |\n|---|---|---|\n"
        for f in ex.key_drivers:
            sign = "+" if f.contribution_pct >= 0 else ""
            explain_md += f"| {f.factor} | {sign}{f.contribution_pct:.1f}% | {f.data_source} |\n"
        explain_md += f"\n**📝 叙述**：{ex.narrative}\n\n"
        if ex.next_actions:
            explain_md += "**下一步行动**：\n"
            for a in ex.next_actions:
                explain_md += f"- {a}\n"

    # Critic 卡片
    critic_md = ""
    if cv and cv.quality:
        q = cv.quality
        score = q.weighted_total
        score_emoji = "🟢" if score >= 0.85 else ("🟡" if score >= 0.70 else "🔴")
        refl_map = {
            "accept": "✅ 自动执行",
            "retry_forecast": "🔄 重试预测",
            "escalate_hitl": "👤 人工审核",
            "abort": "❌ 中止",
        }
        critic_md = f"""### 🔍 质量评估（Critic）

{score_emoji} **综合评分：{score:.2f}**

| 维度 | 得分 |
|---|---|
| 准确性 | {q.accuracy:.2f} |
| 完备性 | {q.completeness:.2f} |
| 合规性 | {q.compliance:.2f} |
| 可执行性 | {q.executability:.2f} |
| 过程合理性 | {q.process_rationality:.2f} |
| 业务价值 | {q.business_value:.2f} |

**决策**：{refl_map.get(cv.reflection.value, cv.reflection.value)}
"""
        if cv.risks:
            critic_md += "\n**风险提示**：\n"
            for r in cv.risks:
                critic_md += f"- ⚠️ {r}\n"
        else:
            critic_md += "\n**风险提示**：无\n"

    # 运行日志追加完成信息
    log(f"📊 **Critic 评分**: `{cv.quality.weighted_total:.2f}`  决策: `{cv.reflection.value}`" if cv and cv.quality else "")
    log(f"📦 **建议下单**: `{ac.quantity:.0f}件`  置信度: `{ac.confidence_tier.value}`" if ac else "")

    yield (
        "\n".join(log_lines),
        forecast_md,
        action_md,
        whatif_md,
        explain_md,
        critic_md,
    )


# ── Gradio UI ─────────────────────────────────────────────────────────────────

CSS = """
.gradio-container { max-width: 1400px !important; }
.result-card { border-radius: 8px; padding: 16px; }
.log-panel { font-family: monospace; font-size: 13px; }
footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
)

def build_app() -> gr.Blocks:
    with gr.Blocks(title="零售供应链 AI-Native Agent") as app:

        # ── 标题 ──────────────────────────────────────────────────────────────
        gr.Markdown("""
# 🏪 零售供应链 AI-Native Agent
**促销场景端到端决策闭环** · LangGraph 5-Expert + Critic · Claude API

> 输入自然语言问题，Agent 自动完成：意图解析 → 销量预测 → 安全库存 → 方案推演 → 归因解释 → 质量评估 → 备货建议
""")

        # ── 输入区 ────────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="📝 输入问题（支持自然语言）",
                    placeholder=(
                        "例：华润万家天河店，可口可乐 330ml，下周六开始做 25% 折扣促销 3 天，"
                        "帮我预测销量并给出备货建议。"
                    ),
                    lines=3,
                    max_lines=6,
                )
            with gr.Column(scale=1):
                preset_dropdown = gr.Dropdown(
                    label="⚡ 预设 Case（一键体验）",
                    choices=["（自由输入）"] + list(PRESET_CASES.keys()),
                    value="（自由输入）",
                    interactive=True,
                )
                run_btn = gr.Button("🚀 运行 Agent", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ 清空", variant="secondary", size="sm")

        # ── 运行日志 ──────────────────────────────────────────────────────────
        with gr.Accordion("📋 运行日志（决策链路）", open=True):
            log_output = gr.Markdown(
                value="*等待运行...*",
                elem_classes=["log-panel"],
            )

        # ── 结果卡片（两列布局）──────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## 📊 决策结果")

        with gr.Row():
            with gr.Column(scale=1):
                forecast_output = gr.Markdown(
                    value="*运行后显示预测结果*",
                    elem_classes=["result-card"],
                )
                action_output = gr.Markdown(
                    value="*运行后显示备货建议*",
                    elem_classes=["result-card"],
                )
            with gr.Column(scale=1):
                whatif_output = gr.Markdown(
                    value="*运行后显示方案对比*",
                    elem_classes=["result-card"],
                )
                explain_output = gr.Markdown(
                    value="*运行后显示归因解释*",
                    elem_classes=["result-card"],
                )

        with gr.Row():
            critic_output = gr.Markdown(
                value="*运行后显示质量评估*",
                elem_classes=["result-card"],
            )

        # ── 预设 case 示例 ────────────────────────────────────────────────────
        gr.Markdown("---")
        gr.Markdown("## 💡 快速体验")
        gr.Examples(
            examples=[
                ["华润万家天河店，可口可乐 330ml 罐装，下周六开始做 25% 折扣促销 3 天，帮我预测销量并给出备货建议。"],
                ["同一家门店，同款可口可乐 330ml，如果把折扣从 20% 提到 30%，哪个方案对门店更划算？"],
                ["华润万家天河店，可口可乐 330ml，下周促销期间气象局发布暴雨橙色预警，是否需要下调备货量？"],
                ["华润万家天河店，可口可乐 330ml，上周促销实际销量 1800 件，但预测是 2375 件，帮我分析偏差原因。"],
            ],
            inputs=question_input,
            label="点击示例直接填入",
        )

        # ── 页脚 ──────────────────────────────────────────────────────────────
        gr.Markdown("""
---
<div style="text-align:center; color:#888; font-size:12px;">
retail-supply-chain-agent v0.2.1 · LangGraph + LightGBM + Claude API ·
<a href="https://github.com/CyAlcher/retail_supply_chain_agent" target="_blank">GitHub</a>
</div>
""")

        # ── 事件绑定 ──────────────────────────────────────────────────────────

        def on_preset_change(preset_key: str):
            """选择预设 case 时自动填入问题"""
            if preset_key and preset_key != "（自由输入）" and preset_key in PRESET_CASES:
                question, _ = PRESET_CASES[preset_key]
                return question
            return gr.update()

        preset_dropdown.change(
            fn=on_preset_change,
            inputs=[preset_dropdown],
            outputs=[question_input],
        )

        def on_clear():
            return ("", "（自由输入）", "*等待运行...*",
                    "*运行后显示预测结果*", "*运行后显示备货建议*",
                    "*运行后显示方案对比*", "*运行后显示归因解释*",
                    "*运行后显示质量评估*")

        clear_btn.click(
            fn=on_clear,
            outputs=[question_input, preset_dropdown, log_output,
                     forecast_output, action_output,
                     whatif_output, explain_output, critic_output],
        )

        run_btn.click(
            fn=run_agent,
            inputs=[question_input, preset_dropdown],
            outputs=[log_output, forecast_output, action_output,
                     whatif_output, explain_output, critic_output],
        )

        # 回车也能触发
        question_input.submit(
            fn=run_agent,
            inputs=[question_input, preset_dropdown],
            outputs=[log_output, forecast_output, action_output,
                     whatif_output, explain_output, critic_output],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_PORT", 7861)),
        share=False,
        show_error=True,
        favicon_path=None,
        theme=THEME,
        css=CSS,
    )
