# models/eda_results.py
"""
EDAResults — output contract of EDAAgent.
Carries all computed analytics metrics passed to ReportWriterAgent.
"""
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class HourlyBin:
    """Transaction volume and fraud statistics for a single hourly window."""
    hour_bin: int
    tx_count: int
    fraud_count: int
    fraud_rate: float           # fraud_count / tx_count


@dataclass
class AmountOutlier:
    """A single transaction flagged as an Amount outlier via IQR method."""
    index: int
    amount: float
    is_fraud: bool              # whether Class == 1 for this row


@dataclass
class FeatureCorrelation:
    """Pearson correlation of a PCA feature with the Class column."""
    feature: str                # e.g. "V14"
    correlation: float          # signed correlation value


@dataclass
class EDAResults:
    """
    Full set of EDA metrics produced by EDAAgent.
    All fields are required; agents must not pass partial results.
    """
    # --- Class distribution ---
    class_distribution_count: Dict[int, int]        # {0: n, 1: n}
    class_distribution_pct: Dict[int, float]        # {0: pct, 1: pct}

    # --- Amount statistics ---
    amount_mean: float
    amount_std: float
    amount_percentiles: Dict[str, float]            # keys: p25, p50, p75, p95, p99

    # --- Outlier detection (IQR-based on Amount) ---
    amount_outliers: List[AmountOutlier]

    # --- Temporal trends ---
    hourly_trend: List[HourlyBin]                   # all hourly bins
    top_fraud_windows: List[HourlyBin]              # top-5 bins by fraud_rate

    # --- Fraud vs. legitimate comparison ---
    fraud_amount_mean: float
    legit_amount_mean: float

    # --- Feature discriminability ---
    top_correlated_features: List[FeatureCorrelation]  # top-10 by |correlation|
