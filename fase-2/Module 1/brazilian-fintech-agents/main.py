# main.py
"""
Brazilian Fintech Agents — Pipeline Orchestrator

Runs the three-agent sequential pipeline:
  1. IngestionAgent  — load, validate, clean
  2. EDAAgent        — explore, analyse, detect anomalies
  3. ReportWriterAgent — generate executive_report.md

Usage:
    python main.py
    python main.py --input ./dataset/creditcard.csv --output ./output
"""
import argparse
import sys

from agents.ingestion_agent import IngestionAgent
from agents.eda_agent import EDAAgent
from agents.report_writer_agent import ReportWriterAgent
from tools.logger import get_logger
from tools.exceptions import SchemaValidationError, DataQualityError

log = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Brazilian Fintech Agents — automated transaction analysis pipeline"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    log.info("=" * 60)
    log.info("Brazilian Fintech Agents — Pipeline starting")
    log.info("Input : %s", args.input)
    log.info("Output: %s", args.output)
    log.info("=" * 60)

    try:
        # ── Agent 1: Ingestion ─────────────────────────────────────────
        log.info("[1/3] IngestionAgent")
        df, summary = IngestionAgent().run(args.input)

        # ── Agent 2: EDA ───────────────────────────────────────────────
        log.info("[2/3] EDAAgent")
        eda = EDAAgent().run(df, summary)

        # ── Agent 3: Report Writer ─────────────────────────────────────
        log.info("[3/3] ReportWriterAgent")
        report_path = ReportWriterAgent(output_dir=args.output).run(summary, eda)

        log.info("=" * 60)
        log.info("Pipeline complete.")
        log.info("Report written to: %s", report_path)
        log.info("=" * 60)

    except FileNotFoundError as exc:
        log.error("Input file not found: %s", exc)
        sys.exit(1)
    except SchemaValidationError as exc:
        log.error("Schema validation failed: %s", exc)
        sys.exit(1)
    except DataQualityError as exc:
        log.error("Data quality check failed: %s", exc)
        sys.exit(1)
    except Exception as exc:
        log.exception("Unexpected error: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
