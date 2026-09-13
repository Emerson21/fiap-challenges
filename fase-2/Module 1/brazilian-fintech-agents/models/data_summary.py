# models/data_summary.py
"""
DataSummary — output contract of IngestionAgent.
Carries basic structural statistics about the loaded CSV.
"""
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DateRange:
    """Represents the temporal span derived from the Time column."""
    min_time_sec: float
    max_time_sec: float
    span_hours: float


@dataclass
class AmountStats:
    """Descriptive statistics for the Amount column."""
    min: float
    max: float
    mean: float
    std: float
    median: float


@dataclass
class DataSummary:
    """
    Structured summary produced by IngestionAgent after loading and
    cleaning the raw transaction CSV. Passed downstream to EDAAgent
    and ReportWriterAgent.
    """
    shape: Tuple[int, int]                  # (rows, cols)
    dtypes: Dict[str, str]                  # col -> dtype name as string
    missing_values: Dict[str, int]          # col -> null count (pre-imputation)
    duplicates_removed: int                 # number of duplicate rows dropped
    date_range: DateRange                   # time span from Time column
    class_distribution: Dict[int, int]      # {0: n_legit, 1: n_fraud}
    amount_stats: AmountStats               # Amount column statistics
