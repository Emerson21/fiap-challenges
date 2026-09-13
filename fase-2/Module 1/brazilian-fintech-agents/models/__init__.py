# models/__init__.py
# New LLM-driven models
from .data_profile import DataProfile, FieldProfile, PrescribedAnalysis
from .analysis_result import AnalysisResult, RiskWindow, FeatureSignal, OutlierRecord

# Legacy models (kept for reference)
from .data_summary import DataSummary, DateRange, AmountStats
from .eda_results import EDAResults, HourlyBin, AmountOutlier, FeatureCorrelation

__all__ = [
    # New
    "DataProfile", "FieldProfile", "PrescribedAnalysis",
    "AnalysisResult", "RiskWindow", "FeatureSignal", "OutlierRecord",
    # Legacy
    "DataSummary", "DateRange", "AmountStats",
    "EDAResults", "HourlyBin", "AmountOutlier", "FeatureCorrelation",
]
