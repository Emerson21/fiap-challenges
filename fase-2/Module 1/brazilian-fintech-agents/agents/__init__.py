# agents/__init__.py
from .data_profiler_agent import DataProfilerAgent
from .analysis_agent import AnalysisAgent
from .report_writer_agent import ReportWriterAgent

# Legacy agents (kept for reference — no longer called from main.py)
from .ingestion_agent import IngestionAgent
from .eda_agent import EDAAgent

__all__ = [
    "DataProfilerAgent",
    "AnalysisAgent",
    "ReportWriterAgent",
    "IngestionAgent",
    "EDAAgent",
]
