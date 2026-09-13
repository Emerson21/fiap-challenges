# models/data_profile.py
"""
DataProfile — output contract of DataProfilerAgent (Agent 1).

Carries the LLM's semantic understanding of the dataset schema,
including per-field relevance to the analysis goal, prescribed analyses,
and identified fraud signal fields.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FieldProfile:
    """Semantic profile of a single dataset column."""
    name: str
    dtype: str
    null_count: int
    description: str                  # LLM-generated explanation of what this field is
    role: str                         # "target" | "temporal_index" | "key_numeric" | "pca_feature" | "unknown"
    relevant_to_goal: bool            # whether this field matters for the analysis goal
    relevance_reason: str             # LLM explanation of why (or why not) it's relevant


@dataclass
class PrescribedAnalysis:
    """A specific analysis the LLM recommends running on this dataset."""
    name: str           # e.g. "class_imbalance", "temporal_trend"
    description: str    # what to compute
    goal_link: str      # why this helps achieve the analysis goal


@dataclass
class DataProfile:
    """
    Full semantic profile of the dataset, produced by DataProfilerAgent.
    Passed downstream to AnalysisAgent and ReportWriterAgent.
    """
    dataset_name: str
    analysis_goal: str                        # e.g. "detect fraudulent transactions"
    shape: Tuple[int, int]                    # (rows, cols)
    fields: List[FieldProfile]                # one entry per column
    target_field: Optional[str]               # column identified as the label/target
    domain: str                               # inferred domain (e.g. "Financial fraud detection")
    fraud_signal_fields: List[str]            # columns most likely to carry the fraud signal
    prescribed_analyses: List[PrescribedAnalysis]  # analyses to run in Agent 2
    llm_summary: str                          # free-text dataset summary from LLM
    generated_by: str = "llm"                 # "llm" | "fallback"

    def relevant_fields(self) -> List[FieldProfile]:
        """Return only fields marked as relevant to the goal."""
        return [f for f in self.fields if f.relevant_to_goal]
