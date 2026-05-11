"""核心 Pydantic 数据契约 — 所有跨层数据结构在此定义"""
from __future__ import annotations
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


# ── 枚举 ──────────────────────────────────────────────────────────────────────

class ConfidenceTier(str, Enum):
    AUTO   = "自动执行"
    REVIEW = "需要复核"
    REJECT = "拒绝"

class ReflectionAction(str, Enum):
    ACCEPT        = "accept"
    RETRY_FORECAST = "retry_forecast"
    RETRY_WHATIF  = "retry_whatif"
    ESCALATE_HITL = "escalate_hitl"
    ABORT         = "abort"

class Scenario(str, Enum):
    PROMO   = "D4_promo"
    SEASONAL = "D1_seasonal_peak"
    DEFAULT = "S_default"


# ── 输入契约 ──────────────────────────────────────────────────────────────────

class PromoFactor(BaseModel):
    discount_rate: float = Field(..., ge=0.0, le=1.0, description="折扣率 0~1")
    start_date: str
    end_date: str
    duration_days: int = Field(..., ge=1)
    coupon_stacked: bool = False

class WeatherFactor(BaseModel):
    avg_temp: float | None = None
    rain_prob: float | None = Field(None, ge=0.0, le=1.0)
    alert_level: str | None = None   # "暴雨橙色" 等

class CalendarFactor(BaseModel):
    is_weekend: bool = False
    is_holiday: bool = False
    holiday_name: str | None = None

class TaskContext(BaseModel):
    task_id: str
    sku_id: str
    store_id: str
    forecast_horizon_days: int = Field(7, ge=1, le=90)
    promo: PromoFactor | None = None
    weather: WeatherFactor | None = None
    calendar: CalendarFactor | None = None
    history_days: int = 90
    raw_question: str = ""
    scenario: Scenario = Scenario.DEFAULT
    tenant_id: str = "default"


# ── 各 Expert 输出契约 ────────────────────────────────────────────────────────

class ForecastResult(BaseModel):
    p25: float
    p50: float
    p75: float
    model_used: str
    feature_importance: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False
    baseline_p50: float | None = None
    mape_vs_baseline: float | None = None

class SafetyStockResult(BaseModel):
    safety_stock_units: float
    coverage_days: float
    service_level: float
    z_score: float
    demand_std: float

class WhatIfScenario(BaseModel):
    label: str
    discount_rate: float
    forecast_p50: float
    gross_profit: float | None = None

class WhatIfResult(BaseModel):
    scenarios: list[WhatIfScenario]
    recommended: str
    recommendation_reason: str

class AttributionFactor(BaseModel):
    factor: str
    contribution_pct: float
    data_source: str

class ExplainResult(BaseModel):
    key_drivers: list[AttributionFactor]
    narrative: str = Field(..., max_length=500)
    next_actions: list[str]

class QualityScore(BaseModel):
    accuracy: float = Field(..., ge=0.0, le=1.0)
    completeness: float = Field(..., ge=0.0, le=1.0)
    compliance: float = Field(..., ge=0.0, le=1.0)
    executability: float = Field(..., ge=0.0, le=1.0)
    process_rationality: float = Field(..., ge=0.0, le=1.0)
    business_value: float = Field(..., ge=0.0, le=1.0)

    @property
    def weighted_total(self) -> float:
        weights = [0.25, 0.15, 0.20, 0.20, 0.10, 0.10]
        scores  = [self.accuracy, self.completeness, self.compliance,
                   self.executability, self.process_rationality, self.business_value]
        return sum(w * s for w, s in zip(weights, scores))

class UncertaintyReport(BaseModel):
    p25: float
    p50: float
    p75: float
    interval_width: float
    confidence_tier: ConfidenceTier
    alert_triggered: bool = False
    alert_reason: str = ""

class CriticVerdict(BaseModel):
    plan_ok: bool = True
    plan_issues: list[str] = Field(default_factory=list)
    quality: QualityScore | None = None
    uncertainty: UncertaintyReport | None = None
    risks: list[str] = Field(default_factory=list)
    reflection: ReflectionAction = ReflectionAction.ACCEPT
    retry_count: int = 0

class AggregatedResult(BaseModel):
    forecast: ForecastResult | None = None
    safety_stock: SafetyStockResult | None = None
    what_if: WhatIfResult | None = None
    explain: ExplainResult | None = None
    critic: CriticVerdict | None = None

class ActionRecommendation(BaseModel):
    action_type: str          # "下单" / "调拨" / "下架"
    quantity: float
    confidence_tier: ConfidenceTier
    rationale: str
    forecast_p50: float
    safety_stock_units: float


# ── LangGraph 状态 ────────────────────────────────────────────────────────────

class PlannerState(BaseModel):
    """LangGraph StateGraph 的状态对象"""
    task: TaskContext
    plan_experts: list[str] = Field(default_factory=list)
    forecast_result: ForecastResult | None = None
    safety_stock_result: SafetyStockResult | None = None
    what_if_result: WhatIfResult | None = None
    explain_result: ExplainResult | None = None
    critic_verdict: CriticVerdict | None = None
    action: ActionRecommendation | None = None
    hitl_required: bool = False
    hitl_approved: bool | None = None
    audit_trail: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    retry_count: int = 0
    llm_call_count: int = 0
    total_tokens: int = 0
