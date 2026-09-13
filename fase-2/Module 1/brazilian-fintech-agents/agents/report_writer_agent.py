# agents/report_writer_agent.py
"""
Agent 3 — ReportWriterAgent

Responsibilities:
  - Receive DataProfile (from Agent 1) + AnalysisResult (from Agent 2)
  - Call Gemini to write the FULL executive report in Markdown
  - If LLM is unavailable, assemble the report from structured data
    using template rendering (graceful fallback)
  - Write the report to output/executive_report.md
"""
import os
from datetime import datetime
from pathlib import Path

from models.data_profile import DataProfile
from models.analysis_result import AnalysisResult
from tools import get_logger
from tools.markdown_helpers import make_table, make_section, make_insight
import tools.gemini_client as llm

log = get_logger(__name__)


class ReportWriterAgent:
    """Agent 3: LLM-authored executive report writer."""

    VERSION = "0.2.0"

    def __init__(self, output_dir: str = "./output"):
        self.output_dir = Path(output_dir)
        self.output_file = "executive_report.md"

    def run(self, profile: DataProfile, analysis: AnalysisResult) -> str:
        """
        Generate and write the executive report.

        Args:
            profile (DataProfile): Dataset understanding from Agent 1.
            analysis (AnalysisResult): Analysis results from Agent 2.

        Returns:
            str: Absolute path to the written report.
        """
        log.info("ReportWriterAgent v%s | source: %s", self.VERSION, analysis.generated_by)

        if llm.is_available():
            content = self._llm_report(profile, analysis)
            source = "llm"
        else:
            log.warning("Gemini not available — using template fallback for report")
            content = self._template_report(profile, analysis)
            source = "fallback"

        content = self._append_footer(content, profile, source)
        output_path = self._write(content)
        log.info("Report written to: %s | source: %s", output_path, source)
        return str(output_path)

    # ------------------------------------------------------------------
    # LLM-authored report
    # ------------------------------------------------------------------

    def _llm_report(self, profile: DataProfile, analysis: AnalysisResult) -> str:
        insights_json = "\n".join(
            f'  {i+1}. Title: "{ins["title"]}" | Finding: "{ins["finding"]}" | Action: "{ins["recommended_action"]}"'
            for i, ins in enumerate(analysis.llm_actionable_insights)
        )

        outlier_rows = "\n".join(
            f"  - Row {o.index}: €{o.amount:.2f} ({'FRAUD' if o.is_fraud else 'Legitimate'})"
            for o in analysis.outliers[:10]
        )

        risk_rows = "\n".join(
            f"  - Hour {w.hour_bin}: {w.fraud_count} fraud / {w.tx_count} tx ({w.fraud_rate_pct:.3f}%)"
            for w in analysis.top_risk_windows
        )

        feature_rows = "\n".join(
            f"  - {f.feature}: r={f.correlation:+.4f}"
            for f in analysis.feature_signals[:5]
        )

        prompt = f"""You are a senior financial analyst writing an executive report for a fintech risk team.

ANALYSIS GOAL: "{profile.analysis_goal}"
DATASET: {profile.dataset_name} — {profile.shape[0]:,} rows × {profile.shape[1]} columns
DOMAIN: {profile.domain}

DATASET PROFILE SUMMARY:
{profile.llm_summary}

KEY STATISTICS:
- Fraud rate: {analysis.fraud_rate_pct:.4f}% ({analysis.class_distribution.get("1", 0):,} fraudulent transactions)
- Average transaction amount: €{analysis.amount_mean:.2f} (std: €{analysis.amount_std:.2f})
- Fraud vs. Legitimate amount: €{analysis.fraud_amount_mean:.2f} vs €{analysis.legit_amount_mean:.2f}
- High-value outliers detected: {len(analysis.outliers)} (threshold: €{analysis.outlier_threshold:.2f})

ANALYTICAL INTERPRETATION:
{analysis.llm_interpretation}

ANOMALY FINDINGS:
{analysis.llm_anomaly_narrative}

KEY FINDINGS:
{chr(10).join(f"- {f}" for f in analysis.llm_key_findings)}

TOP RISK TIME WINDOWS:
{risk_rows}

TOP OUTLIER TRANSACTIONS:
{outlier_rows}

TOP CORRELATED FEATURES:
{feature_rows}

PREPARED INSIGHTS:
{insights_json}

Write a complete, professional executive report in Markdown with EXACTLY these sections:

# Executive Report — [give a title based on the goal]

## 1. Executive Summary
[2 paragraph summary of what was found]

## 2. Dataset Overview
[Include a Markdown table with key stats: rows, columns, fraud count, fraud rate, time span, amount range]

## 3. Transaction Analysis
### 3.1 Amount Distribution
[Table: mean, std, P25, P50, P75, P95, P99]

### 3.2 Temporal Trends
[Table of top risk time windows with hour bin, tx count, fraud count, fraud rate]

### 3.3 Fraud Signals
[Table of top correlated features with correlation values and a brief interpretation]

## 4. Anomalies Detected
### 4.1 High-Value Outlier Transactions
[Table of outlier transactions: row index, amount, classification]

### 4.2 Highest-Risk Time Windows
[Table]

## 5. Actionable Insights
[Write exactly 3 insights using the prepared insights above, formatted as:
### Insight N: [Title]
**Finding:** ...
**Recommended Action:** ...]

## 6. Methodology
[Brief explanation of IQR method, temporal binning, and correlation analysis]

Write the report now. Use the exact data provided. Be specific with numbers.
Do not add any text before or after the Markdown report."""

        content = llm.call(prompt)
        if not content:
            log.warning("LLM report generation failed — falling back to template")
            return self._template_report(profile, analysis)
        return content

    # ------------------------------------------------------------------
    # Template fallback report
    # ------------------------------------------------------------------

    def _template_report(self, profile: DataProfile, analysis: AnalysisResult) -> str:
        n_rows, n_cols = profile.shape
        sections = []

        # Header
        sections.append(
            f"# Executive Report — {profile.analysis_goal.title()}\n\n"
            f"**Dataset:** {profile.dataset_name}  \n"
            f"**Goal:** {profile.analysis_goal}  \n"
            f"**Domain:** {profile.domain}\n"
        )

        # Executive Summary
        fraud_n = analysis.class_distribution.get("1", 0)
        sections.append(
            make_section("1. Executive Summary") +
            f"This report summarises the automated analysis of **{n_rows:,} transactions** "
            f"with the goal of **{profile.analysis_goal}**. "
            f"The pipeline detected **{fraud_n:,} fraudulent transactions** "
            f"({analysis.fraud_rate_pct:.3f}% of total volume).\n\n"
            f"{profile.llm_summary or ''}"
        )

        # Dataset Overview
        overview_rows = [
            ["Total rows", f"{n_rows:,}"],
            ["Total columns", str(n_cols)],
            ["Target field", profile.target_field or "N/A"],
            ["Fraud count", f"{fraud_n:,}"],
            ["Fraud rate", f"{analysis.fraud_rate_pct:.3f}%"],
            ["Amount mean", f"€{analysis.amount_mean:.2f}"],
            ["Amount std", f"€{analysis.amount_std:.2f}"],
        ]
        sections.append(make_section("2. Dataset Overview") + make_table(["Property", "Value"], overview_rows))

        # Analysis
        p = analysis.amount_percentiles
        amount_rows = [
            ["Mean", f"€{analysis.amount_mean:.2f}"],
            ["Std Dev", f"€{analysis.amount_std:.2f}"],
            ["P25", f"€{p.get('p25', 0):.2f}"],
            ["Median (P50)", f"€{p.get('p50', 0):.2f}"],
            ["P75", f"€{p.get('p75', 0):.2f}"],
            ["P95", f"€{p.get('p95', 0):.2f}"],
            ["P99", f"€{p.get('p99', 0):.2f}"],
        ]
        trend_rows = [
            [str(w.hour_bin), f"{w.tx_count:,}", str(w.fraud_count), f"{w.fraud_rate_pct:.3f}%"]
            for w in analysis.top_risk_windows
        ]
        feat_rows = [
            [f.feature, f"{f.correlation:+.4f}", f.interpretation or "—"]
            for f in analysis.feature_signals[:8]
        ]
        sections.append(
            make_section("3. Transaction Analysis") +
            make_section("3.1 Amount Distribution", level=3) + make_table(["Statistic", "Value"], amount_rows) + "\n" +
            make_section("3.2 Temporal Trends — Top Risk Windows", level=3) + make_table(["Hour Bin", "Tx Count", "Fraud Count", "Fraud Rate"], trend_rows) + "\n" +
            make_section("3.3 Fraud Signal Features", level=3) + make_table(["Feature", "Correlation", "Interpretation"], feat_rows)
        )

        # Anomalies
        outlier_rows = [
            [str(o.index), f"€{o.amount:.2f}", "🚨 FRAUD" if o.is_fraud else "Legitimate"]
            for o in analysis.outliers
        ]
        sections.append(make_section("4. Anomalies") + make_table(["Row", "Amount", "Class"], outlier_rows))

        # Insights
        insights_content = make_section("5. Actionable Insights")
        if analysis.llm_actionable_insights:
            for i, ins in enumerate(analysis.llm_actionable_insights[:3], 1):
                insights_content += make_insight(i, ins.get("title", ""), ins.get("finding", ""), ins.get("recommended_action", "")) + "\n"
        else:
            insights_content += "_Insights not available (LLM fallback mode)._\n"
        sections.append(insights_content)

        # Methodology
        sections.append(
            make_section("6. Methodology") +
            "Outlier detection uses the **IQR method** (upper bound = Q3 + 3×IQR). "
            "Temporal analysis bins the time column into 1-hour windows. "
            "Feature correlation uses Pearson correlation with the target variable.\n"
        )

        return "\n\n".join(sections)

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _append_footer(self, content: str, profile: DataProfile, source: str) -> str:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        footer = (
            f"\n\n---\n\n"
            f"_Report generated by **Brazilian Fintech Agents** pipeline_  \n"
            f"_Goal: {profile.analysis_goal}_  \n"
            f"_Run date: {ts}_  \n"
            f"_Dataset: {profile.dataset_name} — {profile.shape[0]:,} rows_  \n"
            f"_Report source: {source}_"
        )
        return content + footer

    def _write(self, content: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / self.output_file
        path.write_text(content, encoding="utf-8")
        return path.resolve()
