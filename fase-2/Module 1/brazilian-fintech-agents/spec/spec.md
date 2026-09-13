# Specification Document — Brazilian Fintech Agents

> **Status:** Final  
> **Version:** 2.0.0  
> **Date:** 2026-09-13  
> **Module:** FIAP — Phase 2, Module 1

---

## 1. Problem Statement

A Brazilian Fintech company currently relies on human analysts to perform monthly analysis of its credit-card transaction database. The manual process involves:

- Consolidating data from multiple sources
- Identifying anomalies and suspicious patterns (e.g., fraud)
- Writing executive reports for stakeholders

This process consumes several analyst-days per month and is error-prone and non-scalable. The goal of this project is to build an autonomous multi-agent pipeline — powered by a Large Language Model — that replaces this manual workflow end-to-end and is capable of analysing **any CSV dataset** given an explicit analysis goal.

---

## 2. Objectives

| # | Objective |
|---|-----------|
| O-1 | Automatically ingest and validate any financial transaction CSV file |
| O-2 | Use an LLM to semantically understand the dataset schema in the context of a user-defined goal |
| O-3 | Prescribe and execute appropriate analyses based on the LLM's understanding |
| O-4 | Detect anomalies and suspicious transactions (fraud signals) using statistical methods |
| O-5 | Generate a complete, LLM-authored executive report in Markdown |
| O-6 | Run the full pipeline with a single command and minimal human intervention |
| O-7 | Operate gracefully without an LLM API key (pandas-only fallback) |

---

## 3. Dataset

### 3.1 Source

| Property | Value |
|---|---|
| File | `./dataset/creditcard.csv` |
| Origin | European cardholders — September 2013 |
| Download | [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Transactions | ~284,807 rows (283,726 after deduplication) |
| Fraud Cases | 473 (0.167% — highly imbalanced) |
| Duration covered | ~48 hours |

> ⚠️ The dataset is **not committed** to the repository (143MB > GitHub 100MB limit). Download and place at `dataset/creditcard.csv`.

### 3.2 Schema

| Column | Type | Description |
|--------|------|-------------|
| `Time` | float64 | Seconds elapsed since the first transaction in the dataset |
| `V1` to `V28` | float64 | Anonymized PCA-transformed features (confidential original variables) |
| `Amount` | float64 | Transaction monetary value (EUR) |
| `Class` | int64 | Target label: `1` = fraud, `0` = legitimate |

### 3.3 Known Data Quality Issues

- **Highly imbalanced classes**: fraud is only 0.167% of data. Standard accuracy is a misleading metric — use **AUPRC** (Area Under Precision-Recall Curve).
- **1,081 duplicate rows** present in the raw dataset — removed by Agent 1.
- No missing values expected in the original dataset, but the pipeline validates defensively.
- `Time` is relative (seconds from first record), not an absolute timestamp.

---

## 4. Required Architecture

The system is a **sequential LLM-driven multi-agent pipeline** implemented in Python. Each agent has a single responsibility. Agents communicate through typed Python dataclasses. The pipeline accepts an analysis goal that drives all LLM reasoning.

```
[creditcard.csv]  +  [--goal "detect fraudulent transactions"]
        │
        ▼
┌───────────────────────────────────────────┐
│  Agent 1 — DataProfilerAgent  (LLM)       │
│  Input:  csv_path + goal                  │
│  Output: DataProfile JSON                 │
└───────────────────────────────────────────┘
        │
        ▼  DataFrame + DataProfile
┌───────────────────────────────────────────┐
│  Agent 2 — AnalysisAgent  (pandas + LLM)  │
│  Input:  DataFrame + DataProfile          │
│  Output: AnalysisResult JSON              │
└───────────────────────────────────────────┘
        │
        ▼  DataProfile + AnalysisResult
┌───────────────────────────────────────────┐
│  Agent 3 — ReportWriterAgent  (LLM)       │
│  Input:  DataProfile + AnalysisResult     │
│  Output: executive_report.md              │
└───────────────────────────────────────────┘
```

---

## 5. Agent Specifications

### 5.1 Agent 1 — `DataProfilerAgent`

**File:** `agents/data_profiler_agent.py`

**Trigger:** Receives `csv_path: str` and `goal: str`.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Load CSV | Read with `pd.read_csv`; detect and remove duplicate rows |
| Null validation | Count nulls per column; raise `DataQualityError` if any column exceeds 20% nulls |
| Data cleaning | Impute numeric nulls with column median; coerce types |
| Schema metadata | Compute: shape, dtypes, null counts, sample rows, `describe()` stats |
| LLM schema analysis | Send schema + goal to Gemini: *"What does each field represent? Which are relevant to the goal? What analyses should be run?"* |
| Output | Return `(DataFrame, DataProfile)` |

**`DataProfile` output contract:**

```json
{
  "dataset_name": "creditcard.csv",
  "analysis_goal": "detect fraudulent transactions",
  "shape": { "rows": 283726, "columns": 31 },
  "fields": [
    {
      "name": "Class",
      "dtype": "int64",
      "null_count": 0,
      "description": "Fraud label — 1 = fraud, 0 = legitimate",
      "role": "target",
      "relevant_to_goal": true,
      "relevance_reason": "This IS the fraud indicator — primary target variable"
    }
  ],
  "target_field": "Class",
  "domain": "Credit card fraud detection",
  "fraud_signal_fields": ["V14", "V12", "V10", "V17", "V11"],
  "prescribed_analyses": [
    {
      "name": "class_imbalance",
      "description": "Count and percentage of each target class",
      "goal_link": "Establishes baseline fraud prevalence"
    }
  ],
  "llm_summary": "This dataset contains anonymised credit-card transactions..."
}
```

**Failure conditions:**
- File not found → raise `FileNotFoundError`
- More than 20% nulls in any column → raise `DataQualityError`

**Fallback:** If `GEMINI_API_KEY` is absent, returns a basic structural profile without LLM field descriptions or prescribed analyses.

---

### 5.2 Agent 2 — `AnalysisAgent`

**File:** `agents/analysis_agent.py`

**Trigger:** Receives `DataFrame` + `DataProfile` from Agent 1.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Read prescribed analyses | Use `DataProfile.prescribed_analyses` to determine what to compute |
| Class distribution | Count and percentage of each target class value |
| Amount statistics | Mean, std, P25/P50/P75/P95/P99 percentiles |
| Outlier detection | IQR method: flag `Amount > Q3 + 3×IQR` (threshold: €293.24) |
| Temporal analysis | Bin `Time` column into 1-hour windows; compute fraud rate per window |
| Feature correlation | Pearson correlation of all numeric features with target column |
| Fraud comparison | Mean `Amount` for fraud vs. legitimate transactions |
| LLM interpretation | Send all computed stats + `DataProfile` to Gemini for contextual narrative, anomaly description, and 3 actionable insights |

**`AnalysisResult` output contract:**

```json
{
  "class_distribution": { "0": 283253, "1": 473 },
  "fraud_rate_pct": 0.1667,
  "amount_mean": 88.47,
  "outlier_threshold": 293.24,
  "outliers": [{ "index": 2, "amount": 378.66, "is_fraud": false }],
  "top_risk_windows": [{ "hour_bin": 26, "fraud_count": 27, "fraud_rate_pct": 1.556 }],
  "feature_signals": [{ "feature": "V17", "correlation": -0.3135, "interpretation": "..." }],
  "llm_interpretation": "...",
  "llm_anomaly_narrative": "...",
  "llm_key_findings": ["...", "..."],
  "llm_actionable_insights": [
    { "title": "...", "finding": "...", "recommended_action": "..." }
  ],
  "generated_by": "llm"
}
```

**Fallback:** If LLM is unavailable, returns all computed pandas stats with empty LLM narrative fields.

---

### 5.3 Agent 3 — `ReportWriterAgent`

**File:** `agents/report_writer_agent.py`

**Trigger:** Receives `DataProfile` + `AnalysisResult` from Agents 1 and 2.

**Responsibilities:**

| Task | Detail |
|------|--------|
| Build context prompt | Combine DataProfile, AnalysisResult stats and LLM narratives into a rich prompt |
| LLM report authoring | Call Gemini with full context; ask it to write the complete executive report in Markdown |
| Fallback rendering | If LLM unavailable, render report from structured data using markdown template helpers |
| Write output | Save to `./output/executive_report.md` with metadata footer |

**Report structure (LLM-authored):**

```markdown
# Executive Report — [LLM-generated title based on goal]
## 1. Executive Summary
## 2. Dataset Overview
## 3. Transaction Analysis
   ### 3.1 Amount Distribution
   ### 3.2 Temporal Trends
   ### 3.3 Fraud Signals
## 4. Anomalies Detected
   ### 4.1 High-Value Outlier Transactions
   ### 4.2 Highest-Risk Time Windows
## 5. Actionable Insights
   ### Insight 1: [LLM-generated title]
   ### Insight 2: ...
   ### Insight 3: ...
## 6. Methodology
---
_Report generated by Brazilian Fintech Agents pipeline_
_Goal: detect fraudulent transactions_
_Report source: llm | fallback_
```

---

## 6. Technical Requirements

| Requirement | Specification |
|-------------|---------------|
| Language | Python 3.10+ |
| Paradigm | Sequential data-flow (no event loops or async queues) |
| Agent orchestration | Each agent is a Python class with a `.run()` method; orchestrator calls them in order |
| LLM Provider | Google Gemini (`gemini-3.6-flash`) via `google-genai` SDK |
| Core Dependencies | `pandas`, `numpy`, `scipy`, `google-genai`, `python-dotenv` |
| API Key | Read from `GEMINI_API_KEY` environment variable or `.env` file — never hardcoded |
| Secrets policy | No API keys, credentials, or PII committed to the repository |
| Output | `executive_report.md` written to `./output/` directory |
| CLI | `python main.py --input <csv> --output <dir> --goal "<goal>"` |
| Fallback | If API key missing, pipeline completes in pandas-only mode |

---

## 7. Deliverables

| # | Deliverable | Status | Location |
|---|-------------|--------|----------|
| D-1 | **Source Code** — public GitHub repo with README | ✅ Complete | Repository root |
| D-2 | **Architecture Diagram** — visual agent flow | ✅ Complete | `design/architecture_diagram.png` |
| D-3 | **Technical Report** (1–2 pages) — architecture decisions + anomaly strategy | ✅ Complete | `technical_report.md` / `technical_report_pt_br.md` |
| D-4 | **`executive_report.md`** — pipeline output on provided dataset | ✅ Complete | `output/executive_report.md` |

---

## 8. Acceptance Criteria

| ID | Criterion | Status |
|----|-----------|--------|
| AC-1 | Agent 1 validates CSV structure and raises descriptive errors for bad input | ✅ |
| AC-2 | Agent 1 handles null values without crashing | ✅ |
| AC-3 | Agent 1 uses LLM to identify field roles and prescribe analyses for the goal | ✅ |
| AC-4 | Agent 2 executes the analyses prescribed by Agent 1 | ✅ |
| AC-5 | Agent 2 produces LLM-interpreted narrative of statistical findings | ✅ |
| AC-6 | Agent 3 produces a valid, complete LLM-authored Markdown report | ✅ |
| AC-7 | The report includes **exactly 3** data-backed actionable insights | ✅ |
| AC-8 | The pipeline runs end-to-end with a single command (`python main.py`) | ✅ |
| AC-9 | The pipeline completes gracefully without a Gemini API key (fallback mode) | ✅ |
| AC-10 | No secrets, keys, or PII are present in the repository | ✅ |
| AC-11 | The pipeline is dataset-agnostic — works on any CSV with a `--goal` argument | ✅ |

---

## 9. Out of Scope

- Real-time transaction monitoring (batch-only)
- Integration with external data sources beyond the input CSV
- Machine learning model training or deployment
- User interface (CLI only)
- Automated scheduling / cron orchestration

---

## 10. Resolved Questions (v1 → v2)

| # | Question | Resolution |
|---|----------|-----------|
| Q-1 | Should the pipeline output intermediate artifacts? | No — single `executive_report.md` output only |
| Q-2 | Customer segment proxy acceptable? | Out of scope — no customer ID available |
| Q-3 | Anomaly table: all rows or top-N? | Top 20 outliers by IQR method |
| Q-4 | Single `main.py` or CLI with arguments? | CLI with `--input`, `--output`, `--goal` |
| Q-5 | Should LLM be used? | Yes — Gemini powers all 3 agents with graceful fallback |
