"""Layer1 感知层：自然语言槽位解析
使用 Claude API tool_use 做结构化槽位抽取，fallback 到规则匹配
"""
from __future__ import annotations
import os
import re
import json
import anthropic
from retail_agent.schemas import TaskContext, PromoFactor, WeatherFactor, CalendarFactor, Scenario

_client: anthropic.Anthropic | None = None

def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    return _client


# ── tool schema ───────────────────────────────────────────────────────────────

_PARSE_TOOL = {
    "name": "parse_retail_query",
    "description": "从零售供应链自然语言问题中提取结构化槽位",
    "input_schema": {
        "type": "object",
        "properties": {
            "sku_id":          {"type": "string", "description": "商品ID，如 SKU_COKE_330ML"},
            "store_id":        {"type": "string", "description": "门店ID，如 STORE_CRW_TH"},
            "discount_rate":   {"type": "number", "description": "折扣率 0~1，如 25% 折扣 → 0.25"},
            "duration_days":   {"type": "integer", "description": "促销天数"},
            "start_date":      {"type": "string", "description": "促销开始日期 YYYY-MM-DD"},
            "avg_temp":        {"type": "number", "description": "平均气温（摄氏度）"},
            "is_weekend":      {"type": "boolean", "description": "是否周末"},
            "scenario":        {"type": "string", "enum": ["promo", "seasonal", "default"],
                                "description": "场景类型"},
            "has_what_if":     {"type": "boolean", "description": "是否需要对比推演"},
        },
        "required": ["scenario"],
    },
}

_SYSTEM_PROMPT = (
    "你是零售供应链 AI 系统的意图解析模块。"
    "从用户输入中提取结构化槽位，调用 parse_retail_query 工具返回结果。"
    "门店名称映射：华润万家天河店→STORE_CRW_TH，其他门店保留原名。"
    "商品映射：可口可乐330ml罐装→SKU_COKE_330ML，其他商品保留原名。"
    "如果信息缺失，使用合理默认值。"
)


def _parse_with_llm(question: str) -> dict:
    """调用 Claude tool_use 解析槽位，返回 dict"""
    client = _get_client()
    resp = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        timeout=30.0,
        system=_SYSTEM_PROMPT,
        tools=[_PARSE_TOOL],
        tool_choice={"type": "tool", "name": "parse_retail_query"},
        messages=[{"role": "user", "content": question}],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name == "parse_retail_query":
            return block.input
    return {}


def _parse_with_rules(question: str) -> dict:
    """规则 fallback，保持原有逻辑"""
    q = question
    discount = 0.0
    m = re.search(r"(\d+)%\s*折扣", q)
    if m:
        discount = int(m.group(1)) / 100.0
    m2 = re.search(r"(\d+)折", q)
    if m2 and not m:
        discount = 1 - int(m2.group(1)) / 10.0

    duration = 3
    m3 = re.search(r"(\d+)\s*天", q)
    if m3:
        duration = int(m3.group(1))

    temp = 30.0
    m4 = re.search(r"(\d+)\s*[℃°]", q)
    if m4:
        temp = float(m4.group(1))

    is_weekend = any(w in q for w in ["周末", "周六", "周日", "星期六", "星期日"])
    has_promo = any(k in q for k in ["促销", "折扣", "打折", "优惠", "活动"]) or discount > 0
    has_what_if = any(k in q for k in ["对比", "推演", "如果", "方案", "哪个"])

    return {
        "sku_id": "SKU_COKE_330ML",
        "store_id": "STORE_CRW_TH",
        "discount_rate": discount,
        "duration_days": duration,
        "start_date": "2026-06-13",
        "avg_temp": temp,
        "is_weekend": is_weekend,
        "scenario": "promo" if has_promo else "default",
        "has_what_if": has_what_if,
    }


def parse(question: str, task_id: str = "TASK-001") -> TaskContext:
    """解析自然语言问题 → TaskContext，优先用 Claude API，失败则 fallback 规则"""
    try:
        slots = _parse_with_llm(question)
    except Exception:
        slots = _parse_with_rules(question)

    discount_rate = float(slots.get("discount_rate") or 0.0)
    duration_days = int(slots.get("duration_days") or 3)
    start_date    = slots.get("start_date") or "2026-06-13"
    avg_temp      = float(slots.get("avg_temp") or 30.0)
    is_weekend    = bool(slots.get("is_weekend", False))
    scenario_str  = slots.get("scenario", "default")
    has_promo     = scenario_str == "promo" or discount_rate > 0

    scenario_map = {
        "promo":    Scenario.PROMO,
        "seasonal": Scenario.SEASONAL,
        "default":  Scenario.DEFAULT,
    }
    scenario = scenario_map.get(scenario_str, Scenario.DEFAULT)

    # 推算结束日期
    from datetime import date, timedelta
    try:
        sd = date.fromisoformat(start_date)
        end_date = (sd + timedelta(days=duration_days - 1)).isoformat()
    except ValueError:
        end_date = start_date

    promo = PromoFactor(
        discount_rate=discount_rate,
        start_date=start_date,
        end_date=end_date,
        duration_days=duration_days,
    ) if has_promo else None

    return TaskContext(
        task_id=task_id,
        sku_id=slots.get("sku_id") or "SKU_COKE_330ML",
        store_id=slots.get("store_id") or "STORE_CRW_TH",
        forecast_horizon_days=duration_days,
        promo=promo,
        weather=WeatherFactor(avg_temp=avg_temp),
        calendar=CalendarFactor(is_weekend=is_weekend),
        scenario=scenario,
        raw_question=question,
    )
