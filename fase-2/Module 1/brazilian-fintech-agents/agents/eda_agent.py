# agents/eda_agent.py
"""
Agent 2 — Exploratory Data Analysis (EDA)

Responsible for:
  - Class distribution (fraud vs. legitimate)
  - Amount statistics and percentile breakdown
  - IQR-based outlier detection on Amount
  - Temporal trend analysis (hourly binning via Time column)
  - Fraud vs. legitimate amount comparison
  - PCA feature correlation with Class label
"""
import pandas as pd
import numpy as np
from typing import List

from models.data_summary import DataSummary
from models.eda_results import (
    EDAResults,
    HourlyBin,
    AmountOutlier,
    FeatureCorrelation,
)
from tools.logger import get_logger

log = get_logger(__name__)

# PCA feature columns (anonymised by dataset provider)
V_FEATURES = [f"V{i}" for i in range(1, 29)]


class EDAAgent:
    """Performs full exploratory analysis on the cleaned transaction dataframe."""

    VERSION = "0.1.0"

    TOP_FRAUD_WINDOWS = 5       # number of top fraud-rate hours to surface
    TOP_FEATURES = 10           # number of top correlated V-features to report
    IQR_MULTIPLIER = 3.0        # outlier = Q3 + (k * IQR)
    MAX_OUTLIER_ROWS = 20       # cap the anomaly list for the report

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, df: pd.DataFrame, summary: DataSummary) -> EDAResults:
        """
        Execute all EDA steps and return structured results.

        Args:
            df (pd.DataFrame): Cleaned dataframe from IngestionAgent.
            summary (DataSummary): Metadata from IngestionAgent.

        Returns:
            EDAResults: Full set of computed analytics metrics.
        """
        log.info("EDAAgent v%s started — rows: %d", self.VERSION, len(df))

        class_count, class_pct = self._class_distribution(df)
        amount_mean, amount_std, percentiles = self._amount_stats(df)
        outliers = self._detect_amount_outliers(df)
        hourly_trend = self._compute_hourly_trend(df)
        fraud_mean, legit_mean = self._fraud_amount_comparison(df)
        top_windows = self._top_fraud_windows(hourly_trend)
        top_features = self._feature_correlations(df)

        log.info(
            "EDAAgent complete — fraud: %d (%.3f%%) | outliers: %d",
            class_count.get(1, 0),
            class_pct.get(1, 0.0) * 100,
            len(outliers),
        )

        return EDAResults(
            class_distribution_count=class_count,
            class_distribution_pct=class_pct,
            amount_mean=amount_mean,
            amount_std=amount_std,
            amount_percentiles=percentiles,
            amount_outliers=outliers,
            hourly_trend=hourly_trend,
            top_fraud_windows=top_windows,
            fraud_amount_mean=fraud_mean,
            legit_amount_mean=legit_mean,
            top_correlated_features=top_features,
        )

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _class_distribution(self, df: pd.DataFrame):
        """Count and percentage split of fraud vs. legitimate transactions."""
        counts = df["Class"].value_counts().to_dict()
        total = len(df)
        pct = {k: round(v / total, 6) for k, v in counts.items()}
        log.info(
            "Class distribution — legit: %d | fraud: %d",
            counts.get(0, 0),
            counts.get(1, 0),
        )
        return counts, pct

    def _amount_stats(self, df: pd.DataFrame):
        """Mean, std, and key percentiles of the Amount column."""
        mean = float(df["Amount"].mean())
        std = float(df["Amount"].std())
        percentiles = {
            "p25": float(df["Amount"].quantile(0.25)),
            "p50": float(df["Amount"].quantile(0.50)),
            "p75": float(df["Amount"].quantile(0.75)),
            "p95": float(df["Amount"].quantile(0.95)),
            "p99": float(df["Amount"].quantile(0.99)),
        }
        log.info("Amount stats — mean: %.2f | std: %.2f | p99: %.2f", mean, std, percentiles["p99"])
        return mean, std, percentiles

    def _detect_amount_outliers(self, df: pd.DataFrame) -> List[AmountOutlier]:
        """
        IQR-based outlier detection on the Amount column.

        Strategy: upper_bound = Q3 + (IQR_MULTIPLIER * IQR)
        Rationale: robust to skew; no normality assumption required.
        """
        q1 = df["Amount"].quantile(0.25)
        q3 = df["Amount"].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + self.IQR_MULTIPLIER * iqr

        flagged = df[df["Amount"] > upper_bound].head(self.MAX_OUTLIER_ROWS)
        outliers = [
            AmountOutlier(
                index=int(idx),
                amount=float(row["Amount"]),
                is_fraud=bool(row["Class"] == 1),
            )
            for idx, row in flagged.iterrows()
        ]
        log.info(
            "Outlier detection — Q3: %.2f | IQR: %.2f | bound: %.2f | flagged: %d",
            q3,
            iqr,
            upper_bound,
            len(outliers),
        )
        return outliers

    def _compute_hourly_trend(self, df: pd.DataFrame) -> List[HourlyBin]:
        """
        Bin the Time column into 1-hour windows and compute
        transaction volume and fraud count per bin.
        """
        df = df.copy()
        df["hour_bin"] = (df["Time"] // 3600).astype(int)

        grouped = df.groupby("hour_bin").agg(
            tx_count=("Class", "count"),
            fraud_count=("Class", "sum"),
        ).reset_index()

        trend = [
            HourlyBin(
                hour_bin=int(row["hour_bin"]),
                tx_count=int(row["tx_count"]),
                fraud_count=int(row["fraud_count"]),
                fraud_rate=round(float(row["fraud_count"]) / float(row["tx_count"]), 6),
            )
            for _, row in grouped.iterrows()
        ]
        log.info("Hourly trend computed — %d bins", len(trend))
        return trend

    def _fraud_amount_comparison(self, df: pd.DataFrame):
        """Mean transaction Amount for fraud vs. legitimate transactions."""
        fraud_mean = float(df[df["Class"] == 1]["Amount"].mean())
        legit_mean = float(df[df["Class"] == 0]["Amount"].mean())
        log.info(
            "Amount comparison — fraud mean: %.2f | legit mean: %.2f",
            fraud_mean,
            legit_mean,
        )
        return fraud_mean, legit_mean

    def _top_fraud_windows(self, hourly_trend: List[HourlyBin]) -> List[HourlyBin]:
        """Return the top-N hourly bins sorted by fraud_rate descending."""
        sorted_bins = sorted(hourly_trend, key=lambda b: b.fraud_rate, reverse=True)
        return sorted_bins[: self.TOP_FRAUD_WINDOWS]

    def _feature_correlations(self, df: pd.DataFrame) -> List[FeatureCorrelation]:
        """
        Compute Pearson correlation of each V1–V28 feature with Class.
        Return the top-N features sorted by absolute correlation value.
        """
        corr_series = df[V_FEATURES].corrwith(df["Class"])
        top = corr_series.abs().nlargest(self.TOP_FEATURES)
        result = [
            FeatureCorrelation(
                feature=feat,
                correlation=round(float(corr_series[feat]), 6),
            )
            for feat in top.index
        ]
        log.info(
            "Top correlated feature: %s (r=%.4f)",
            result[0].feature if result else "N/A",
            result[0].correlation if result else 0.0,
        )
        return result
