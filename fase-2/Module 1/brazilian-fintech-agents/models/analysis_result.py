# models/analysis_result.py
"""
AnalysisResult — output contract of AnalysisAgent (Agent 2).

Contains computed statistics (from pandas/numpy) plus LLM-generated
narrative interpretation of those statistics in the context of the
analysis goal.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RiskWindow:
    """A time window with elevated fraud concentration."""
    hour_bin: int
    tx_count: int
    fraud_count: int
    fraud_rate_pct: float


@dataclass
class FeatureSignal:
    """A dataset feature and its correlation with the target."""
    feature: str
    correlation: float
    interpretation: str   # LLM explanation of why this feature matters


@dataclass
class OutlierRecord:
    """A single transaction flagged as a statistical outlier."""
    index: int
    amount: float
    is_fraud: bool


@dataclass
class AnalysisResult:
    """
    Full analysis output from AnalysisAgent, combining computed statistics
    with LLM-generated interpretation.
    """
    # --- Class / target distribution ---
    class_distribution: Dict[str, int]       # {"0": n_legit, "1": n_fraud}
    fraud_rate_pct: float

    # --- Amount statistics ---
    amount_mean: float
    amount_std: float
    amount_percentiles: Dict[str, float]     # p25, p50, p75, p95, p99
    outlier_threshold: float                 # IQR upper bound
    outliers: List[OutlierRecord]

    # --- Temporal patterns ---
    risk_windows: List[RiskWindow]           # all hourly bins
    top_risk_windows: List[RiskWindow]       # top-5 by fraud_rate

    # --- Feature signals ---
    feature_signals: List[FeatureSignal]     # top correlated features

    # --- Fraud comparison ---
    fraud_amount_mean: float
    legit_amount_mean: float

    # --- LLM-generated narratives ---
    llm_interpretation: str                  # overall dataset interpretation
    llm_anomaly_narrative: str              # description of anomalies found
    llm_key_findings: List[str]             # 3-5 bullet findings
    llm_actionable_insights: List[Dict]     # list of {title, finding, recommended_action}

    generated_by: str = "llm"              # "llm" | "fallback"
