# agents/analysis_agent.py
"""
Agent 2 — AnalysisAgent

Responsibilities:
  - Read the DataProfile from Agent 1 (which analyses to run and why)
  - Execute each prescribed analysis using pandas/numpy
  - Call Gemini to interpret the computed statistics IN THE CONTEXT of the goal
  - Return a fully populated AnalysisResult with both computed stats
    and LLM-generated narrative (interpretation, anomaly descriptions, insights)

Fallback: if Gemini is unavailable, returns pandas-only stats with
          no LLM narrative.
"""
import json
import pandas as pd
import numpy as np
from typing import List, Tuple

from models.data_profile import DataProfile
from models.analysis_result import AnalysisResult, RiskWindow, FeatureSignal, OutlierRecord
from tools import get_logger
import tools.gemini_client as llm

log = get_logger(__name__)

IQR_MULTIPLIER = 3.0
MAX_OUTLIERS = 20
TOP_WINDOWS = 5
TOP_FEATURES = 10


class AnalysisAgent:
    """
    Agent 2: Executes analyses prescribed by DataProfilerAgent and uses
    Gemini to interpret findings in the context of the analysis goal.
    """

    VERSION = "0.2.0"

    def run(self, df: pd.DataFrame, profile: DataProfile) -> AnalysisResult:
        """
        Execute prescribed analyses and return structured results.

        Args:
            df (pd.DataFrame): Cleaned DataFrame from DataProfilerAgent.
            profile (DataProfile): Schema understanding + analysis prescription from Agent 1.

        Returns:
            AnalysisResult: Computed stats + LLM narrative interpretations.
        """
        log.info(
            "AnalysisAgent v%s | goal: '%s' | analyses: %d",
            self.VERSION,
            profile.analysis_goal,
            len(profile.prescribed_analyses),
        )

        # Identify key columns from profile
        target = self._resolve_target(df, profile)
        amount_col = self._resolve_amount_col(df, profile)
        time_col = self._resolve_time_col(df, profile)
        feature_cols = self._resolve_feature_cols(df, profile, target)

        log.info("Resolved → target: %s | amount: %s | time: %s | features: %d",
                 target, amount_col, time_col, len(feature_cols))

        # --- Execute prescribed analyses (pandas/numpy) ---
        class_dist, fraud_rate = self._class_distribution(df, target)
        amount_mean, amount_std, percentiles = self._amount_stats(df, amount_col)
        outlier_threshold, outliers = self._detect_outliers(df, amount_col, target)
        risk_windows = self._temporal_trend(df, time_col, target) if time_col else []
        top_windows = sorted(risk_windows, key=lambda w: w.fraud_rate_pct, reverse=True)[:TOP_WINDOWS]
        feature_signals_raw = self._feature_correlations(df, feature_cols, target)
        fraud_mean, legit_mean = self._fraud_amount_comparison(df, amount_col, target)

        # --- LLM interpretation ---
        stats_summary = self._build_stats_summary(
            class_dist, fraud_rate, amount_mean, amount_std, percentiles,
            outliers, outlier_threshold, top_windows, feature_signals_raw,
            fraud_mean, legit_mean,
        )
        llm_result = self._call_llm_interpretation(profile, stats_summary)

        # --- Build feature signals with LLM interpretations ---
        llm_feature_map = {f["feature"]: f.get("interpretation", "") for f in llm_result.get("feature_signals", [])}
        feature_signals = [
            FeatureSignal(
                feature=feat,
                correlation=corr,
                interpretation=llm_feature_map.get(feat, ""),
            )
            for feat, corr in feature_signals_raw
        ]

        generated_by = "llm" if llm.is_available() and llm_result.get("llm_interpretation") else "fallback"

        result = AnalysisResult(
            class_distribution=class_dist,
            fraud_rate_pct=fraud_rate,
            amount_mean=amount_mean,
            amount_std=amount_std,
            amount_percentiles=percentiles,
            outlier_threshold=outlier_threshold,
            outliers=outliers,
            risk_windows=risk_windows,
            top_risk_windows=top_windows,
            feature_signals=feature_signals,
            fraud_amount_mean=fraud_mean,
            legit_amount_mean=legit_mean,
            llm_interpretation=llm_result.get("llm_interpretation", "LLM unavailable."),
            llm_anomaly_narrative=llm_result.get("llm_anomaly_narrative", ""),
            llm_key_findings=llm_result.get("llm_key_findings", []),
            llm_actionable_insights=llm_result.get("llm_actionable_insights", []),
            generated_by=generated_by,
        )

        log.info(
            "AnalysisAgent complete — fraud: %.3f%% | outliers: %d | insights: %d | source: %s",
            fraud_rate,
            len(outliers),
            len(result.llm_actionable_insights),
            generated_by,
        )
        return result

    # ------------------------------------------------------------------
    # Column resolution helpers
    # ------------------------------------------------------------------

    def _resolve_target(self, df: pd.DataFrame, profile: DataProfile) -> str:
        if profile.target_field and profile.target_field in df.columns:
            return profile.target_field
        # Heuristic fallback
        for candidate in ["Class", "class", "label", "Label", "fraud", "Fraud", "target"]:
            if candidate in df.columns:
                return candidate
        # Last resort: last column
        return df.columns[-1]

    def _resolve_amount_col(self, df: pd.DataFrame, profile: DataProfile) -> str:
        for f in profile.fields:
            if f.role == "key_numeric" and f.name in df.columns:
                return f.name
        for candidate in ["Amount", "amount", "value", "Value", "price", "Price"]:
            if candidate in df.columns:
                return candidate
        # First float column that isn't target
        target = profile.target_field or ""
        for col in df.select_dtypes(include="float64").columns:
            if col != target:
                return col
        return df.columns[0]

    def _resolve_time_col(self, df: pd.DataFrame, profile: DataProfile) -> str:
        for f in profile.fields:
            if f.role == "temporal_index" and f.name in df.columns:
                return f.name
        for candidate in ["Time", "time", "timestamp", "Timestamp", "date", "Date"]:
            if candidate in df.columns:
                return candidate
        return None

    def _resolve_feature_cols(self, df: pd.DataFrame, profile: DataProfile, target: str) -> List[str]:
        relevant = [f.name for f in profile.relevant_fields() if f.name in df.columns and f.name != target]
        if relevant:
            return relevant
        return [c for c in df.select_dtypes(include="number").columns if c != target]

    # ------------------------------------------------------------------
    # Prescribed analyses (pandas/numpy)
    # ------------------------------------------------------------------

    def _class_distribution(self, df: pd.DataFrame, target: str) -> Tuple[dict, float]:
        counts = df[target].value_counts().to_dict()
        counts = {str(k): int(v) for k, v in counts.items()}
        n = len(df)
        fraud_count = counts.get("1", 0)
        fraud_rate = round(fraud_count / n * 100, 4) if n else 0.0
        return counts, fraud_rate

    def _amount_stats(self, df: pd.DataFrame, amount_col: str) -> Tuple[float, float, dict]:
        s = df[amount_col]
        percentiles = {
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
        }
        return float(s.mean()), float(s.std()), percentiles

    def _detect_outliers(self, df: pd.DataFrame, amount_col: str, target: str) -> Tuple[float, List[OutlierRecord]]:
        q1 = df[amount_col].quantile(0.25)
        q3 = df[amount_col].quantile(0.75)
        iqr = q3 - q1
        threshold = q3 + IQR_MULTIPLIER * iqr
        flagged = df[df[amount_col] > threshold].head(MAX_OUTLIERS)
        outliers = [
            OutlierRecord(
                index=int(idx),
                amount=float(row[amount_col]),
                is_fraud=bool(row.get(target, 0) == 1),
            )
            for idx, row in flagged.iterrows()
        ]
        log.info("Outlier detection — bound: %.2f | flagged: %d", threshold, len(outliers))
        return float(threshold), outliers

    def _temporal_trend(self, df: pd.DataFrame, time_col: str, target: str) -> List[RiskWindow]:
        df = df.copy()
        df["_hour_bin"] = (df[time_col] // 3600).astype(int)
        grouped = df.groupby("_hour_bin").agg(
            tx_count=(target, "count"),
            fraud_count=(target, "sum"),
        ).reset_index()
        return [
            RiskWindow(
                hour_bin=int(row["_hour_bin"]),
                tx_count=int(row["tx_count"]),
                fraud_count=int(row["fraud_count"]),
                fraud_rate_pct=round(float(row["fraud_count"]) / float(row["tx_count"]) * 100, 4),
            )
            for _, row in grouped.iterrows()
        ]

    def _feature_correlations(self, df: pd.DataFrame, feature_cols: List[str], target: str) -> List[Tuple[str, float]]:
        valid = [c for c in feature_cols if c in df.columns and c != target]
        if not valid:
            return []
        corr = df[valid].corrwith(df[target])
        top = corr.abs().nlargest(TOP_FEATURES)
        return [(feat, round(float(corr[feat]), 6)) for feat in top.index]

    def _fraud_amount_comparison(self, df: pd.DataFrame, amount_col: str, target: str) -> Tuple[float, float]:
        fraud_mean = float(df[df[target] == 1][amount_col].mean()) if (df[target] == 1).any() else 0.0
        legit_mean = float(df[df[target] == 0][amount_col].mean()) if (df[target] == 0).any() else 0.0
        return fraud_mean, legit_mean

    # ------------------------------------------------------------------
    # LLM interpretation
    # ------------------------------------------------------------------

    def _build_stats_summary(
        self, class_dist, fraud_rate, amount_mean, amount_std, percentiles,
        outliers, outlier_threshold, top_windows, feature_signals, fraud_mean, legit_mean,
    ) -> str:
        top_w = "\n".join(
            f"  - Hour {w.hour_bin}: {w.fraud_count} fraud / {w.tx_count} total ({w.fraud_rate_pct:.3f}%)"
            for w in top_windows[:3]
        )
        top_f = "\n".join(
            f"  - {feat}: r = {corr:+.4f}"
            for feat, corr in feature_signals[:5]
        )
        return f"""CLASS DISTRIBUTION: {class_dist}
FRAUD RATE: {fraud_rate:.4f}%
AMOUNT — mean: {amount_mean:.2f} | std: {amount_std:.2f} | p99: {percentiles.get('p99', 0):.2f}
OUTLIER THRESHOLD (IQR 3x): {outlier_threshold:.2f} | flagged: {len(outliers)} transactions
FRAUD vs. LEGIT AMOUNT — fraud mean: {fraud_mean:.2f} | legit mean: {legit_mean:.2f}
TOP RISK TIME WINDOWS:
{top_w}
TOP CORRELATED FEATURES:
{top_f}"""

    def _call_llm_interpretation(self, profile: DataProfile, stats_summary: str) -> dict:
        if not llm.is_available():
            return {}

        prompt = f"""You are a senior financial analyst. Your goal is: "{profile.analysis_goal}"

You have analyzed a dataset with the following profile:
DOMAIN: {profile.domain}
DATASET: {profile.dataset_name} — {profile.shape[0]:,} rows
TARGET VARIABLE: {profile.target_field}
LLM DATASET SUMMARY: {profile.llm_summary}

COMPUTED STATISTICS:
{stats_summary}

Based on this information, provide a complete analytical interpretation in JSON:

{{
  "llm_interpretation": "<2-3 paragraph overall interpretation of what these statistics reveal about {profile.analysis_goal}>",
  "llm_anomaly_narrative": "<1-2 paragraph description of the anomalies and risk patterns detected>",
  "llm_key_findings": [
    "<finding 1 — specific, data-backed>",
    "<finding 2>",
    "<finding 3>",
    "<finding 4 — optional>",
    "<finding 5 — optional>"
  ],
  "llm_actionable_insights": [
    {{
      "title": "<short insight title>",
      "finding": "<what the data shows, citing specific numbers>",
      "recommended_action": "<concrete business action the fintech risk team should take>"
    }},
    {{...}},
    {{...}}
  ],
  "feature_signals": [
    {{
      "feature": "<feature name>",
      "interpretation": "<what this feature's correlation with fraud means in business terms>"
    }}
  ]
}}

IMPORTANT: Generate EXACTLY 3 actionable insights. Each must cite specific numbers from the data.
Respond ONLY with valid JSON. No extra text."""

        result = llm.call_json(prompt)
        return result if result else {}
