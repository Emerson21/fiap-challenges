# agents/data_profiler_agent.py
"""
Agent 1 — DataProfilerAgent

Responsibilities:
  - Load and validate the CSV file (pandas)
  - Compute structural metadata (shape, dtypes, nulls, sample rows)
  - Call Gemini with the analysis GOAL + schema to:
      * Describe what each field represents
      * Identify which fields are relevant to the goal and why
      * Identify the target variable
      * Prescribe which analyses Agent 2 should execute
      * Identify expected fraud signal fields
  - Return a fully populated DataProfile dataclass

Fallback: if Gemini is unavailable, returns a basic profile
          with pandas-derived metadata and no LLM narratives.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

from models.data_profile import DataProfile, FieldProfile, PrescribedAnalysis
from tools import get_logger
from tools.exceptions import SchemaValidationError, DataQualityError
import tools.gemini_client as llm

log = get_logger(__name__)

_PROFILE_SCHEMA = """
{
  "dataset_name": "<filename>",
  "analysis_goal": "<the goal passed in>",
  "domain": "<inferred domain, e.g. Financial fraud detection>",
  "target_field": "<column name of the target/label, or null>",
  "fraud_signal_fields": ["<col1>", "<col2>", "..."],
  "llm_summary": "<2-3 sentence dataset summary in the context of the goal>",
  "fields": [
    {
      "name": "<column name>",
      "dtype": "<pandas dtype>",
      "null_count": 0,
      "description": "<what this field represents>",
      "role": "<target|temporal_index|key_numeric|pca_feature|identifier|unknown>",
      "relevant_to_goal": true,
      "relevance_reason": "<why this field matters for the goal, or why it does not>"
    }
  ],
  "prescribed_analyses": [
    {
      "name": "<analysis_key>",
      "description": "<what to compute>",
      "goal_link": "<how this analysis helps achieve the goal>"
    }
  ]
}
"""


class DataProfilerAgent:
    """
    Agent 1: Uses LLM to semantically understand any CSV dataset
    in the context of a user-defined analysis goal.
    """

    VERSION = "0.2.0"
    NULL_THRESHOLD_PCT = 0.20
    SAMPLE_ROWS = 5

    def run(self, csv_path: str, goal: str) -> Tuple[pd.DataFrame, DataProfile]:
        """
        Load, profile, and semantically understand the dataset.

        Args:
            csv_path (str): Path to the CSV file.
            goal (str): The analysis objective (e.g. "detect fraudulent transactions").

        Returns:
            Tuple[pd.DataFrame, DataProfile]:
                - Cleaned DataFrame
                - DataProfile with LLM semantic understanding
        """
        log.info("DataProfilerAgent v%s | goal: '%s'", self.VERSION, goal)

        df = self._load(csv_path)
        self._validate_nulls(df)
        df = self._clean(df)

        structural_meta = self._compute_structural_meta(df, csv_path)
        profile = self._build_profile(df, structural_meta, goal, csv_path)

        log.info(
            "DataProfilerAgent complete — shape: %s | target: %s | analyses prescribed: %d | source: %s",
            df.shape,
            profile.target_field,
            len(profile.prescribed_analyses),
            profile.generated_by,
        )
        return df, profile

    # ------------------------------------------------------------------
    # Data loading & cleaning
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")
        log.info("Loading CSV: %s", csv_path)
        df = pd.read_csv(path)
        log.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def _validate_nulls(self, df: pd.DataFrame) -> None:
        null_pcts = df.isnull().mean()
        violations = null_pcts[null_pcts > self.NULL_THRESHOLD_PCT]
        if not violations.empty:
            details = ", ".join(f"'{c}': {p:.1%}" for c, p in violations.items())
            raise DataQualityError(f"Null threshold exceeded — {details}")

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            log.warning("Removed %d duplicate rows", removed)
        # Coerce numeric columns
        for col in df.select_dtypes(include=["object"]).columns:
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
        df = df.fillna(df.median(numeric_only=True))
        return df

    # ------------------------------------------------------------------
    # Structural metadata (pandas only — no LLM)
    # ------------------------------------------------------------------

    def _compute_structural_meta(self, df: pd.DataFrame, csv_path: str) -> dict:
        sample = df.head(self.SAMPLE_ROWS).to_dict(orient="list")
        stats = df.describe(include="all").to_dict()

        columns_meta = []
        for col in df.columns:
            col_sample = sample.get(col, [])
            col_sample_str = ", ".join(str(v) for v in col_sample[:3])
            columns_meta.append({
                "name": col,
                "dtype": str(df[col].dtype),
                "null_count": int(df[col].isnull().sum()),
                "sample_values": col_sample_str,
                "unique_count": int(df[col].nunique()),
            })

        return {
            "filename": Path(csv_path).name,
            "shape": df.shape,
            "columns_meta": columns_meta,
        }

    # ------------------------------------------------------------------
    # LLM profile building
    # ------------------------------------------------------------------

    def _build_profile(
        self,
        df: pd.DataFrame,
        meta: dict,
        goal: str,
        csv_path: str,
    ) -> DataProfile:
        if not llm.is_available():
            log.warning("Gemini not available — using fallback DataProfile")
            return self._fallback_profile(df, meta, goal)

        prompt = self._build_prompt(meta, goal)
        raw = llm.call_json(prompt)

        if raw is None:
            log.warning("Gemini returned no response — using fallback DataProfile")
            return self._fallback_profile(df, meta, goal)

        return self._parse_profile(raw, df, goal)

    def _build_prompt(self, meta: dict, goal: str) -> str:
        col_lines = "\n".join(
            f"  - {c['name']} | dtype: {c['dtype']} | nulls: {c['null_count']} "
            f"| unique values: {c['unique_count']} | sample: [{c['sample_values']}]"
            for c in meta["columns_meta"]
        )

        return f"""You are a senior data analyst. Your mission is:

GOAL: "{goal}"

You have been given a CSV dataset named "{meta['filename']}" with {meta['shape'][0]:,} rows and {meta['shape'][1]} columns.

COLUMN SCHEMA AND SAMPLE DATA:
{col_lines}

Given the mission above, answer ALL of the following:
1. What does each column represent in plain language?
2. Which columns are relevant to the goal "{goal}" and WHY?
3. Which column is the target variable (label) for this goal, if any?
4. What specific analyses should the next agent run to achieve the goal?
   For each analysis, explain exactly what to compute and why it helps.
5. Which columns are the strongest expected fraud/anomaly signal fields?
6. Write a 2-3 sentence summary of this dataset in the context of the goal.

Respond ONLY with a valid JSON object matching this EXACT schema (no extra text):
{_PROFILE_SCHEMA}"""

    def _parse_profile(self, raw: dict, df: pd.DataFrame, goal: str) -> DataProfile:
        fields = [
            FieldProfile(
                name=f.get("name", ""),
                dtype=f.get("dtype", ""),
                null_count=int(f.get("null_count", 0)),
                description=f.get("description", ""),
                role=f.get("role", "unknown"),
                relevant_to_goal=bool(f.get("relevant_to_goal", False)),
                relevance_reason=f.get("relevance_reason", ""),
            )
            for f in raw.get("fields", [])
        ]

        analyses = [
            PrescribedAnalysis(
                name=a.get("name", ""),
                description=a.get("description", ""),
                goal_link=a.get("goal_link", ""),
            )
            for a in raw.get("prescribed_analyses", [])
        ]

        return DataProfile(
            dataset_name=raw.get("dataset_name", Path(df.columns[0]).name),
            analysis_goal=goal,
            shape=df.shape,
            fields=fields,
            target_field=raw.get("target_field"),
            domain=raw.get("domain", "Unknown"),
            fraud_signal_fields=raw.get("fraud_signal_fields", []),
            prescribed_analyses=analyses,
            llm_summary=raw.get("llm_summary", ""),
            generated_by="llm",
        )

    # ------------------------------------------------------------------
    # Fallback (no LLM)
    # ------------------------------------------------------------------

    def _fallback_profile(self, df: pd.DataFrame, meta: dict, goal: str) -> DataProfile:
        fields = [
            FieldProfile(
                name=c["name"],
                dtype=c["dtype"],
                null_count=c["null_count"],
                description=f"Column '{c['name']}' with dtype {c['dtype']}",
                role="unknown",
                relevant_to_goal=True,
                relevance_reason="All fields included (fallback mode — LLM unavailable)",
            )
            for c in meta["columns_meta"]
        ]

        default_analyses = [
            PrescribedAnalysis("class_imbalance", "Count target class distribution", "Measures fraud prevalence"),
            PrescribedAnalysis("temporal_trend", "Bin time column into hourly windows", "Identifies fraud time patterns"),
            PrescribedAnalysis("feature_correlation", "Correlate all numeric features with target", "Surfaces fraud predictors"),
            PrescribedAnalysis("amount_outlier_detection", "Apply IQR outlier detection on amount column", "Flags extreme transactions"),
            PrescribedAnalysis("fraud_amount_comparison", "Compare mean amount by class", "Quantifies fraud vs. legit difference"),
        ]

        return DataProfile(
            dataset_name=meta["filename"],
            analysis_goal=goal,
            shape=df.shape,
            fields=fields,
            target_field=None,
            domain="Unknown (fallback mode)",
            fraud_signal_fields=[],
            prescribed_analyses=default_analyses,
            llm_summary="LLM unavailable — basic structural profile only.",
            generated_by="fallback",
        )
