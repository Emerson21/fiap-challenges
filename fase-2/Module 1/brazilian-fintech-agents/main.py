# main.py
"""
Brazilian Fintech Agents — Pipeline Orchestrator (v2 — LLM-driven)

Three-agent LLM pipeline:
  1. DataProfilerAgent  — understand CSV schema in context of the analysis goal
  2. AnalysisAgent      — execute prescribed analyses + LLM interpretation
  3. ReportWriterAgent  — LLM-authored executive report

Usage:
    python main.py
    python main.py --input ./dataset/creditcard.csv --output ./output
    python main.py --input ./dataset/creditcard.csv --goal "detect money laundering"

Environment:
    GEMINI_API_KEY  — required for LLM mode (set in .env or shell)
    GEMINI_MODEL    — optional, defaults to gemini-2.0-flash
"""
import argparse
import sys

from agents.data_profiler_agent import DataProfilerAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_writer_agent import ReportWriterAgent
from tools.logger import get_logger
from tools.exceptions import SchemaValidationError, DataQualityError
import tools.gemini_client as llm

log = get_logger(__name__)

DEFAULT_GOAL = "detect fraudulent transactions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brazilian Fintech Agents — LLM-driven transaction analysis pipeline"
    )
    parser.add_argument(
        "--input",
        default="./dataset/creditcard.csv",
        help="Path to the input CSV file (default: ./dataset/creditcard.csv)",
    )
    parser.add_argument(
        "--output",
        default="./output",
        help="Directory where executive_report.md will be written (default: ./output)",
    )
    parser.add_argument(
        "--goal",
        default=DEFAULT_GOAL,
        help=f'Analysis goal (default: "{DEFAULT_GOAL}")',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Brazilian Fintech Agents v2 — LLM Pipeline starting")
    log.info("Input : %s", args.input)
    log.info("Output: %s", args.output)
    log.info("Goal  : %s", args.goal)
    log.info("LLM   : %s", "Gemini ✓" if llm.is_available() else "⚠️  NOT CONFIGURED — fallback mode")
    log.info("=" * 60)

    try:
        # ── Agent 1: DataProfilerAgent ─────────────────────────────────
        log.info("[1/3] DataProfilerAgent — schema understanding")
        df, profile = DataProfilerAgent().run(args.input, goal=args.goal)

        log.info("  Target field    : %s", profile.target_field)
        log.info("  Domain          : %s", profile.domain)
        log.info("  Signal fields   : %s", profile.fraud_signal_fields[:5])
        log.info("  Analyses        : %d prescribed", len(profile.prescribed_analyses))

        # ── Agent 2: AnalysisAgent ─────────────────────────────────────
        log.info("[2/3] AnalysisAgent — compute + LLM interpretation")
        analysis = AnalysisAgent().run(df, profile)

        log.info("  Fraud rate      : %.4f%%", analysis.fraud_rate_pct)
        log.info("  Outliers        : %d", len(analysis.outliers))
        log.info("  Insights        : %d LLM-generated", len(analysis.llm_actionable_insights))

        # ── Agent 3: ReportWriterAgent ─────────────────────────────────
        log.info("[3/3] ReportWriterAgent — authoring executive report")
        report_path = ReportWriterAgent(output_dir=args.output).run(profile, analysis)

        log.info("=" * 60)
        log.info("Pipeline complete.")
        log.info("Report : %s", report_path)
        log.info("=" * 60)

    except FileNotFoundError as exc:
        log.error("Input file not found: %s", exc)
        sys.exit(1)
    except (SchemaValidationError, DataQualityError) as exc:
        log.error("Data validation failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
