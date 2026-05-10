"""Layer1 感知层：自然语言槽位解析
TODO(真实化): 替换为 Claude API tool_use slot-filling
"""
from __future__ import annotations
import re
from retail_agent.schemas import TaskContext, PromoFactor, WeatherFactor, CalendarFactor, Scenario


_PROMO_KEYWORDS = ["促销", "折扣", "打折", "优惠", "活动"]
_WHATIF_KEYWORDS = ["对比", "推演", "如果", "方案", "哪个"]
_EXPLAIN_KEYWORDS = ["归因", "原因", "为什么", "偏差", "分析"]


def parse(question: str, task_id: str = "TASK-001") -> TaskContext:
    """
    规则匹配解析自然语言问题 → TaskContext
    TODO(真实化): 用 Claude API structured output 做槽位抽取
    """
    q = question

    # 折扣率提取
    discount = 0.0
    m = re.search(r"(\d+)%\s*折扣", q)
    if m:
        discount = int(m.group(1)) / 100.0
    m2 = re.search(r"(\d+)折", q)
    if m2 and not m:
        discount = 1 - int(m2.group(1)) / 10.0

    # 促销天数
    duration = 3
    m3 = re.search(r"(\d+)\s*天", q)
    if m3:
        duration = int(m3.group(1))

    # 温度
    temp = 30.0
    m4 = re.search(r"(\d+)\s*[℃°]", q)
    if m4:
        temp = float(m4.group(1))

    # 是否周末
    is_weekend = any(w in q for w in ["周末", "周六", "周日", "星期六", "星期日"])

    # 场景判断
    has_promo = any(k in q for k in _PROMO_KEYWORDS) or discount > 0
    scenario = Scenario.PROMO if has_promo else Scenario.DEFAULT

    promo = PromoFactor(
        discount_rate=discount,
        start_date="2026-06-13",   # TODO(真实化): 从问题中提取日期
        end_date="2026-06-15",
        duration_days=duration,
    ) if has_promo else None

    return TaskContext(
        task_id=task_id,
        sku_id="SKU_COKE_330ML",       # TODO(真实化): 从问题中提取 SKU
        store_id="STORE_CRW_TH",       # TODO(真实化): 从问题中提取门店
        forecast_horizon_days=duration,
        promo=promo,
        weather=WeatherFactor(avg_temp=temp),
        calendar=CalendarFactor(is_weekend=is_weekend),
        scenario=scenario,
        raw_question=question,
    )
