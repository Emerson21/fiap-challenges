# Specification Document — Brazilian Fintech Agents

> **Status:** Draft  
> **Version:** 0.1.0  
> **Date:** 2026-09-13  
> **Module:** FIAP — Phase 2, Module 1  

---

## 1. Problem Statement

A Brazilian Fintech company currently relies on human analysts to perform monthly analysis of its credit-card transaction database. The manual process involves:

- Consolidating data from multiple sources
- Identifying anomalies and suspicious patterns (e.g., fraud)
- Writing executive reports for stakeholders

This process consumes several analyst-days per month and is error-prone and non-scalable. The goal of this project is to build an autonomous multi-agent pipeline that replaces this manual workflow end-to-end.

---

## 2. Objectives

| # | Objective |
|---|-----------|
| O-1 | Automatically ingest and validate a financial transaction CSV file |
| O-2 | Perform a full Exploratory Data Analysis (EDA) to surface key patterns |
| O-3 | Detect anomalies and suspicious transactions (fraud signals) |
| O-4 | Generate a structured executive report in Markdown format |
| O-5 | Run the full pipeline with minimal human intervention |

---

## 3. Dataset

### 3.1 Source

| Property | Value |
|---|---|
| File | `./dataset/creditcard.csv` |
| Origin | European cardholders — September 2013 (Kaggle public dataset) |
| Total Transactions | ~284,807 rows |
| Fraud Cases | ~492 (approx 0.172% of total — highly imbalanced) |
| Duration covered | 2 days |

### 3.2 Schema

| Column | Type | Description |
|--------|------|-------------|
| `Time` | float64 | Seconds elapsed since the first transaction in the dataset |
| `V1` to `V28` | float64 | Anonymized PCA-transformed features (confidential original variables) |
| `Amount` | float64 | Transaction monetary value (EUR) |
| `Class` | int64 | Target label: `1` = fraud, `0` = legitimate |

### 3.3 Known Data Quality Issues

- **Highly imbalanced classes**: fraud is only 0.172% of data. Standard accuracy is a misleading metric — use **AUPRC** (Area Under Precision-Recall Curve).
- **No missing values** expected in the original dataset, but the pipeline must still validate and handle them defensively.
- **No categorical or temporal string columns** — all fields are numerical.
- `Time` is relative (seconds from first record), not an absolute timestamp.

---

## 4. Required Architecture

The system is a **composed multi-agent pipeline** implemented in Python. Each agent has a single, well-defined responsibility. Agents communicate by passing structured data artifacts (e.g., DataFrames, dictionaries, Markdown strings) through a sequential data flow.

```
 [CSV File]
     |
     v
+--------------+
|  Agent 1     |  Ingestion & Validation
|  Ingestion   |---> Cleaned DataFrame + Summary Stats
+--------------+
     |
     v
+--------------+
|  Agent 2     |  Exploratory Data Analysis
|  EDA         |---> Analysis Results Dict (distributions, trends, anomalies)
+--------------+
     |
     v
+--------------+
|  Agent 3     |  Report Generation
|  Report      |---> executive_report.md
|  Writer      |
+--------------+
```

---

## 5. Agent Specifications

### 5.1 Agent 1 — Ingestion & Validation

**Trigger:** Receives the path to `creditcard.csv`.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Schema validation | Assert expected columns are present (`Time`, `V1`–`V28`, `Amount`, `Class`) |
| Row integrity check | Detect duplicate rows; log and drop them |
| Null / missing values | Count nulls per column; apply appropriate imputation strategy (e.g., median fill for numeric) or drop if threshold exceeded |
| Data type coercion | Ensure `Class` is `int`, `Amount` and `Time` are `float64` |
| Basic summary | Compute and return a `DataSummary` struct |

**Output — `DataSummary`:**

```python
{
  "shape": (rows, cols),
  "dtypes": { col: dtype, ... },
  "missing_values": { col: count, ... },
  "duplicates_removed": int,
  "date_range": {         # derived from Time column
    "min_time_sec": float,
    "max_time_sec": float,
    "span_hours": float
  },
  "class_distribution": { 0: int, 1: int },
  "amount_stats": { "min": float, "max": float, "mean": float, "std": float }
}
```

**Failure conditions:**
- Missing required columns → raise `SchemaValidationError`
- More than 20% nulls in any column → raise `DataQualityError`

---

### 5.2 Agent 2 — Exploratory Data Analysis (EDA)

**Trigger:** Receives the cleaned DataFrame + `DataSummary` from Agent 1.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Class distribution | Count and percentage of fraud vs. legitimate transactions |
| Amount distribution | Histogram buckets, mean/median/std, percentile breakdown (P25, P75, P95, P99) |
| Temporal trends | Convert `Time` (seconds) into hourly and weekly bins; compute transaction volume and fraud rate per bin |
| Top active customers | Since there is no explicit customer ID, proxy by grouping similar `Amount` + `V` feature clusters; report top-N transaction groups by volume |
| Fraud detection signals | Flag transactions where `Class == 1`; compute average `Amount` for fraud vs. legitimate; identify time windows with elevated fraud rates |
| Statistical anomalies | Apply IQR-based outlier detection on `Amount`; flag values beyond `Q3 + 3xIQR` |
| Correlation snapshot | Compute correlation of each `V1`–`V28` feature with `Class` to surface most discriminant features |

**Output — `EDAResults`:**

```python
{
  "class_distribution": { "count": {...}, "pct": {...} },
  "amount_stats": { "mean": float, "std": float, "percentiles": {...} },
  "amount_outliers": [ { "index": int, "amount": float }, ... ],
  "hourly_trend": [ { "hour_bin": int, "tx_count": int, "fraud_count": int }, ... ],
  "fraud_amount_comparison": { "fraud_mean": float, "legit_mean": float },
  "top_fraud_time_windows": [ { "hour_bin": int, "fraud_rate": float }, ... ],
  "top_correlated_features": [ { "feature": str, "correlation": float }, ... ]
}
```

---

### 5.3 Agent 3 — Report Writer

**Trigger:** Receives `DataSummary` + `EDAResults` from Agents 1 and 2.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Executive summary | 1–2 paragraph overview of dataset size, time span, fraud prevalence |
| Analysis section | Structured breakdown of EDA findings (volume, amounts, temporal patterns) |
| Anomalies table | Markdown table listing the top-N outlier/fraud transactions |
| Actionable insights | Exactly 3 insights, each with: title, finding, and recommended action |
| Report footer | Metadata: pipeline run date, data file used, agent versions |

**Output:** `executive_report.md` written to `./output/` directory.

**Report structure:**

```markdown
# Executive Report — Financial Transaction Analysis
## 1. Executive Summary
## 2. Dataset Overview
## 3. Transaction Analysis
### 3.1 Volume & Distribution
### 3.2 Temporal Trends
### 3.3 Fraud Signals
## 4. Anomalies Table
## 5. Actionable Insights
### Insight 1: ...
### Insight 2: ...
### Insight 3: ...
## 6. Methodology Notes
---
_Report generated by Brazilian Fintech Agents pipeline_
```

---

## 6. Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| Language | Python 3.10+ |
| Paradigm | Sequential data-flow (no event loops or async queues required) |
| Agent orchestration | Each agent is a Python class or module with a `.run()` method; the orchestrator calls them in order |
| Core Dependencies | `pandas`, `numpy`, `scipy` (stats), optionally `scikit-learn` (IQR / PCA helpers) |
| No LLM required | The report is generated programmatically — no API keys or external AI services |
| Secrets policy | No API keys, credentials, or personal data committed to the repository |
| Output | `executive_report.md` written to `./output/` directory |

---

## 7. Deliverables

| # | Deliverable | Description |
|---|-------------|-------------|
| D-1 | **Source Code** | Public GitHub repository. Must include `README.md` with run instructions |
| D-2 | **Architecture Diagram** | Visual diagram of the agent flow (e.g., PNG or embedded Mermaid in README) |
| D-3 | **Technical Report** (1–2 pages) | Architecture decisions + justification for anomaly detection strategy |
| D-4 | **`executive_report.md`** | The actual output produced by the pipeline on the provided dataset |

---

## 8. Acceptance Criteria

| ID | Criterion |
|----|-----------|
| AC-1 | Agent 1 validates schema and raises a descriptive error for bad input |
| AC-2 | Agent 1 handles null values without crashing |
| AC-3 | Agent 2 computes all required EDA metrics without manual intervention |
| AC-4 | Agent 3 produces a valid, readable Markdown report |
| AC-5 | The report includes **exactly 3** actionable insights |
| AC-6 | The pipeline runs end-to-end with a single command (e.g., `python main.py`) |
| AC-7 | No secrets, keys, or PII are present in the repository |
| AC-8 | The output report accurately reflects the dataset statistics |

---

## 9. Out of Scope

- Real-time transaction monitoring (batch-only)
- Integration with external data sources or APIs
- Machine learning model training or deployment
- User interface (CLI only)
- Automated scheduling / cron orchestration

---

## 10. Open Questions

| # | Question | Owner |
|---|----------|-------|
| Q-1 | Should the pipeline output intermediate artifacts (e.g., cleaned CSV, EDA JSON)? | Team |
| Q-2 | Is a customer segment proxy acceptable given no explicit customer ID in the dataset? | Team |
| Q-3 | Should the anomaly table include ALL flagged rows or just the top-N? If top-N, what N? | Team |
| Q-4 | Is a single `main.py` entry point sufficient, or is a CLI with arguments (e.g., `--input`, `--output`) preferred? | Team |
