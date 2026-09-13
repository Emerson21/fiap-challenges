# agents/report_writer_agent.py
"""
Agent 3 — Report Writer

Responsible for:
  - Assembling the full executive Markdown report from structured inputs
  - Rendering all sections: summary, dataset overview, analysis, anomalies,
    actionable insights, and methodology notes
  - Writing the final report to the output directory
"""
import os
from datetime import datetime
from pathlib import Path

from models.data_summary import DataSummary
from models.eda_results import EDAResults
from tools.markdown_helpers import make_section, make_table, make_insight
from tools.logger import get_logger

log = get_logger(__name__)


class ReportWriterAgent:
    """Assembles and writes the executive Markdown report."""

    VERSION = "0.1.0"

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_file = "executive_report.md"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, summary: DataSummary, eda: EDAResults) -> str:
        """
        Render the full report and write it to disk.

        Args:
            summary (DataSummary): From IngestionAgent.
            eda (EDAResults): From EDAAgent.

        Returns:
            str: Absolute path to the written report file.
        """
        log.info("ReportWriterAgent v%s started", self.VERSION)

        sections = [
            self._render_header(),
            self._render_executive_summary(summary, eda),
            self._render_dataset_overview(summary),
            self._render_transaction_analysis(eda),
            self._render_anomalies_table(eda),
            self._render_insights(summary, eda),
            self._render_methodology(eda),
            self._render_footer(summary),
        ]

        content = "\n\n".join(sections)
        output_path = self._write_report(content)

        log.info("Report written to: %s", output_path)
        return str(output_path)

    # ------------------------------------------------------------------
    # Section renderers
    # ------------------------------------------------------------------

    def _render_header(self) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "# Executive Report — Financial Transaction Analysis\n\n"
            f"**Generated:** {ts}  \n"
            f"**Pipeline version:** ReportWriterAgent v{self.VERSION}\n"
        )

    def _render_executive_summary(self, summary: DataSummary, eda: EDAResults) -> str:
        n_rows, _ = summary.shape
        fraud_count = eda.class_distribution_count.get(1, 0)
        fraud_pct = eda.class_distribution_pct.get(1, 0.0) * 100
        span_h = summary.date_range.span_hours

        body = (
            f"This report summarises the automated analysis of **{n_rows:,} credit-card transactions** "
            f"spanning **{span_h:.1f} hours** of activity. "
            f"The pipeline detected **{fraud_count:,} fraudulent transactions** "
            f"({fraud_pct:.3f}% of total volume), consistent with a severely imbalanced dataset "
            f"requiring specialised anomaly detection techniques.\n\n"
            f"Key findings, outlier alerts, and three actionable business recommendations are "
            f"detailed in the sections below."
        )

        return make_section("1. Executive Summary") + body

    def _render_dataset_overview(self, summary: DataSummary) -> str:
        dr = summary.date_range
        a = summary.amount_stats
        n_rows, n_cols = summary.shape

        overview_rows = [
            ["Total rows", f"{n_rows:,}"],
            ["Total columns", str(n_cols)],
            ["Duplicates removed", str(summary.duplicates_removed)],
            ["Time span", f"{dr.span_hours:.2f} hours"],
            ["Min time (sec)", f"{dr.min_time_sec:,.0f}"],
            ["Max time (sec)", f"{dr.max_time_sec:,.0f}"],
            ["Amount min", f"€{a.min:.2f}"],
            ["Amount max", f"€{a.max:.2f}"],
            ["Amount mean", f"€{a.mean:.2f}"],
            ["Amount std dev", f"€{a.std:.2f}"],
        ]

        legit = summary.class_distribution.get(0, 0)
        fraud = summary.class_distribution.get(1, 0)
        class_rows = [
            ["Legitimate (0)", f"{legit:,}", f"{legit/n_rows*100:.3f}%"],
            ["Fraudulent (1)", f"{fraud:,}", f"{fraud/n_rows*100:.3f}%"],
        ]

        return (
            make_section("2. Dataset Overview")
            + make_table(["Property", "Value"], overview_rows)
            + "\n"
            + make_section("Class Balance", level=3)
            + make_table(["Class", "Count", "Percentage"], class_rows)
        )

    def _render_transaction_analysis(self, eda: EDAResults) -> str:
        # Amount distribution
        p = eda.amount_percentiles
        amount_rows = [
            ["Mean", f"€{eda.amount_mean:.2f}"],
            ["Std Dev", f"€{eda.amount_std:.2f}"],
            ["P25", f"€{p['p25']:.2f}"],
            ["Median (P50)", f"€{p['p50']:.2f}"],
            ["P75", f"€{p['p75']:.2f}"],
            ["P95", f"€{p['p95']:.2f}"],
            ["P99", f"€{p['p99']:.2f}"],
        ]

        # Temporal — top-10 hourly bins by volume
        sorted_by_vol = sorted(eda.hourly_trend, key=lambda b: b.tx_count, reverse=True)[:10]
        trend_rows = [
            [str(b.hour_bin), f"{b.tx_count:,}", str(b.fraud_count), f"{b.fraud_rate*100:.3f}%"]
            for b in sorted_by_vol
        ]

        # Fraud comparison
        comp_rows = [
            ["Fraudulent", f"€{eda.fraud_amount_mean:.2f}"],
            ["Legitimate", f"€{eda.legit_amount_mean:.2f}"],
            ["Difference", f"€{abs(eda.fraud_amount_mean - eda.legit_amount_mean):.2f}"],
        ]

        # Top correlated features
        feat_rows = [
            [f.feature, f"{f.correlation:+.4f}"]
            for f in eda.top_correlated_features
        ]

        return (
            make_section("3. Transaction Analysis")
            + make_section("3.1 Amount Distribution", level=3)
            + make_table(["Statistic", "Value"], amount_rows)
            + "\n"
            + make_section("3.2 Temporal Trends — Top 10 Busiest Hours", level=3)
            + make_table(["Hour Bin", "Tx Count", "Fraud Count", "Fraud Rate"], trend_rows)
            + "\n"
            + make_section("3.3 Fraud vs. Legitimate — Average Amount", level=3)
            + make_table(["Segment", "Average Amount"], comp_rows)
            + "\n"
            + make_section("3.4 Top Discriminant Features (V1–V28 × Class)", level=3)
            + make_table(["Feature", "Correlation with Class"], feat_rows)
        )

    def _render_anomalies_table(self, eda: EDAResults) -> str:
        if not eda.amount_outliers:
            body = "_No Amount outliers detected above the IQR threshold._\n"
            return make_section("4. Anomalies Table") + body

        rows = [
            [
                str(o.index),
                f"€{o.amount:.2f}",
                "🚨 FRAUD" if o.is_fraud else "Legitimate",
            ]
            for o in eda.amount_outliers
        ]

        # Top fraud-rate windows
        window_rows = [
            [str(b.hour_bin), f"{b.tx_count:,}", str(b.fraud_count), f"{b.fraud_rate*100:.3f}%"]
            for b in eda.top_fraud_windows
        ]

        return (
            make_section("4. Anomalies")
            + make_section("4.1 High-Value Outlier Transactions (IQR method)", level=3)
            + make_table(["Row Index", "Amount", "Class"], rows)
            + "\n"
            + make_section("4.2 Highest-Risk Time Windows", level=3)
            + make_table(["Hour Bin", "Tx Count", "Fraud Count", "Fraud Rate"], window_rows)
        )

    def _render_insights(self, summary: DataSummary, eda: EDAResults) -> str:
        fraud_pct = eda.class_distribution_pct.get(1, 0.0) * 100
        top_window = eda.top_fraud_windows[0] if eda.top_fraud_windows else None
        top_feature = eda.top_correlated_features[0] if eda.top_correlated_features else None

        insight_1 = make_insight(
            1,
            "Targeted Fraud Monitoring During Peak-Risk Hours",
            (
                f"Hour bin **{top_window.hour_bin}** shows the highest fraud rate "
                f"({top_window.fraud_rate*100:.2f}%) in the dataset."
                if top_window else "A time window with elevated fraud was detected."
            ),
            "Deploy real-time transaction scoring during identified peak-risk hours and "
            "trigger additional authentication steps (e.g., OTP) for transactions in those windows.",
        )

        insight_2 = make_insight(
            2,
            "Fraud Prevalence Requires Precision-Recall Optimisation",
            (
                f"Fraud accounts for only **{fraud_pct:.3f}%** of all transactions. "
                "Standard accuracy metrics are misleading; a naive classifier predicting 'legitimate' "
                "would score >99% accuracy while missing all fraud."
            ),
            "Adopt AUPRC (Area Under Precision-Recall Curve) as the primary model evaluation metric. "
            "Apply class-weight balancing or oversampling (e.g., SMOTE) if a ML model is introduced.",
        )

        insight_3 = make_insight(
            3,
            f"Leverage Feature {top_feature.feature if top_feature else 'V-series'} for Fraud Signals",
            (
                f"**{top_feature.feature}** has the strongest correlation with fraud "
                f"(r = {top_feature.correlation:+.4f}), making it the most discriminant PCA component."
                if top_feature else "PCA features show varying correlation with fraud labels."
            ),
            f"Prioritise {top_feature.feature if top_feature else 'high-correlation features'} in any "
            "rule-based or ML-based fraud detection model. Investigate its original business interpretation "
            "with the data provider to design targeted business rules.",
        )

        return (
            make_section("5. Actionable Insights")
            + insight_1
            + "\n"
            + insight_2
            + "\n"
            + insight_3
        )

    def _render_methodology(self, eda: EDAResults) -> str:
        body = (
            "### Anomaly Detection — IQR Method\n\n"
            "Outliers in the `Amount` column are identified using the **Interquartile Range (IQR)** method:\n\n"
            "```\n"
            "Q1 = Amount.quantile(0.25)\n"
            "Q3 = Amount.quantile(0.75)\n"
            "IQR = Q3 - Q1\n"
            "upper_bound = Q3 + 3.0 × IQR\n"
            "flagged = transactions where Amount > upper_bound\n"
            "```\n\n"
            "**Rationale:** IQR is robust to skewed distributions, which is appropriate for transaction amounts "
            "that are heavily right-skewed. The 3× multiplier reduces false positives on legitimate high-value transactions.\n\n"
            "### Temporal Analysis\n\n"
            "The `Time` column (seconds elapsed since first transaction) is binned into 1-hour windows "
            "using integer division (`Time // 3600`). Each bin is analysed for fraud concentration.\n\n"
            "### Feature Correlation\n\n"
            "Pearson correlation between each anonymised PCA feature (V1–V28) and the `Class` label "
            "is computed to surface the most discriminant signals for fraud detection."
        )

        return make_section("6. Methodology Notes") + body

    def _render_footer(self, summary: DataSummary) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return (
            "---\n\n"
            "_Report generated by **Brazilian Fintech Agents** pipeline_  \n"
            f"_Run date: {ts}_  \n"
            f"_Dataset shape: {summary.shape[0]:,} rows × {summary.shape[1]} columns_"
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_report(self, content: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / self.output_file
        output_path.write_text(content, encoding="utf-8")
        return output_path.resolve()
