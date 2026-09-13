# Brazilian Fintech Agents

Automated multi-agent pipeline for financial transaction analysis.
Built for FIAP — Phase 2, Module 1.

## Overview

A three-agent sequential data-flow pipeline that ingests a credit-card transaction CSV, performs exploratory data analysis, detects anomalies, and produces an executive Markdown report — with no manual intervention.

```
[CSV] → IngestionAgent → EDAAgent → ReportWriterAgent → executive_report.md
```

## Project Structure

```
brazilian-fintech-agents/
├── agents/
│   ├── ingestion_agent.py       # Agent 1 — load, validate, clean
│   ├── eda_agent.py             # Agent 2 — analyse, detect anomalies
│   └── report_writer_agent.py   # Agent 3 — generate Markdown report
├── models/
│   ├── data_summary.py          # DataSummary dataclass (Agent 1 output)
│   └── eda_results.py           # EDAResults dataclass (Agent 2 output)
├── tools/
│   ├── exceptions.py            # SchemaValidationError, DataQualityError
│   ├── logger.py                # Centralised logging
│   └── markdown_helpers.py      # Markdown table & section builders
├── dataset/
│   └── creditcard.csv           # Input data (not committed if large)
├── spec/
│   └── spec.md                  # Detailed requirements specification
├── design/
│   └── design.md                # Architecture & class design document
├── output/                      # Generated reports (git-ignored)
├── main.py                      # Pipeline entry point
└── requirements.txt
```

## Requirements

- Python 3.10+
- Dependencies: `pandas`, `numpy`, `scipy`

## Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/fiap-challenges.git
cd "fiap-challenges/fase-2/Module 1/brazilian-fintech-agents"

# Create and activate a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# .venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
# Run with default paths
python main.py

# Run with custom paths
python main.py --input ./dataset/creditcard.csv --output ./output
```

The report will be written to `./output/executive_report.md`.

## Dataset

The pipeline expects `./dataset/creditcard.csv` — a credit-card fraud detection dataset with the following schema:

| Column | Description |
|--------|-------------|
| `Time` | Seconds elapsed since the first transaction |
| `V1`–`V28` | Anonymised PCA-transformed features |
| `Amount` | Transaction value (EUR) |
| `Class` | Label: `1` = fraud, `0` = legitimate |

## Architecture

See [`design/design.md`](design/design.md) for the full architecture diagram and class design.

## Agents

| Agent | File | Responsibility |
|-------|------|----------------|
| **IngestionAgent** | `agents/ingestion_agent.py` | Validate schema, handle nulls, coerce types, produce `DataSummary` |
| **EDAAgent** | `agents/eda_agent.py` | Compute distributions, detect outliers, analyse temporal trends |
| **ReportWriterAgent** | `agents/report_writer_agent.py` | Render and write `executive_report.md` |

## License

Academic project — FIAP. No licence for commercial use.
