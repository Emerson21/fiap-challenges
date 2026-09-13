# Technical Report — Brazilian Fintech Agents

**Project:** FIAP — Phase 2, Module 1  
**Author:** Brazilian Fintech Agents Team  
**Date:** 2026-09-13  
**Version:** 2.0.0

---

## 1. Overview

This report describes the technical decisions made when designing and implementing the **Brazilian Fintech Agents** v2 pipeline — an autonomous multi-agent system powered by **Google Gemini** that ingests credit-card transaction data, semantically understands the dataset schema in the context of a user-defined goal, performs exploratory data analysis, and generates a fully LLM-authored executive report — without human intervention.

The system processes the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 31 features, September 2013), which is characterised by severe class imbalance: fraudulent transactions represent only 0.172% of the total volume.

**Key design principle:** The pipeline is **dataset-agnostic**. By receiving an explicit analysis goal (`--goal`), it can analyse any CSV file without hardcoded domain knowledge.

---

## 2. Architecture Decisions

### 2.1 Sequential Data-Flow over Event-Driven Architecture

**Decision:** The pipeline uses a strict linear, sequential data-flow model: each agent runs to completion and passes its output directly to the next agent via the orchestrator.

**Justification:** The workload is a batch analytics job — there is no need for parallelism, event queues, or asynchronous processing. A sequential flow is:

- **Simpler to reason about** — each step is deterministic and inspectable
- **Easier to debug** — failures are immediately attributed to a specific agent
- **Sufficient for the scale** — the dataset fits in memory (~150MB); no distributed processing is required

An event-driven or async architecture would add complexity (race conditions, message brokers) with no benefit for a single-machine batch job.

---

### 2.2 Mission-Driven Agent 1: Goal-Aware Schema Understanding

**Decision:** `DataProfilerAgent` (Agent 1) receives both the CSV path and an explicit **analysis goal** string (e.g. `"detect fraudulent transactions"`). The LLM prompt is built around this goal, making the agent reason about the dataset through the lens of the mission.

**Justification:** This is the core innovation of v2. Instead of hardcoding which columns are relevant or which analyses to run, Agent 1 asks Gemini:

1. *What does each field represent?*
2. *Which fields are relevant to the goal and why?*
3. *What analyses should be run to achieve the goal?*
4. *What fraud signals should the next agent look for?*

The result is a `DataProfile` JSON that contains `prescribed_analyses` — a list of analysis steps with explicit `goal_link` explanations. **Agent 2 reads this list and executes exactly what Agent 1 prescribed**, not a hardcoded set of functions.

This means the same pipeline can analyse a medical dataset with goal `"detect anomalous patient outcomes"` or a logistics dataset with goal `"identify delayed shipments"` without any code changes.

---

### 2.3 Three-Agent Separation of Concerns

**Decision:** The pipeline is split into three specialised agents — `DataProfilerAgent`, `AnalysisAgent`, and `ReportWriterAgent` — each with a single, isolated responsibility.

**Justification:** Follows the **Single Responsibility Principle**:

| Agent | Owns | Does not own |
|-------|------|-------------|
| DataProfilerAgent | Schema understanding, goal alignment, analysis prescription | Statistical computation, reporting |
| AnalysisAgent | Statistical execution (pandas), LLM interpretation | Data I/O, report formatting |
| ReportWriterAgent | Report authoring (LLM), file I/O | Any computation |

Benefits:
- **Independent testability** — each agent can be unit-tested with mock inputs
- **Replaceability** — any agent can be swapped without affecting others
- **Clarity** — failures in schema understanding vs. analysis vs. reporting are immediately distinguishable in logs

---

### 2.4 LLM Integration via Google Gemini

**Decision:** All three agents use **Google Gemini** (`gemini-3.6-flash`) for their reasoning tasks. Gemini is called via the `google-genai` SDK with structured JSON prompts.

**Justification:** LLM is used where programmatic logic is insufficient:

| Task | Why LLM is needed |
|------|-------------------|
| Field description and role assignment | Requires semantic understanding of column names and sample values — not achievable with regex or dtype inspection alone |
| Prescribing analyses for any dataset | Requires reasoning about the goal in relation to the data — cannot be hardcoded without domain knowledge |
| Interpreting statistical findings | "V17 has r=-0.3135" requires explanation: *"strong inverse correlation means V17 is the primary predictive signal"* |
| Authoring the executive report | Natural language report generation cannot be done with string templates without losing contextual quality |

**Model choice — `gemini-3.6-flash`:**
- Fast inference (~20s per call) vs. `gemini-2.5-pro` (~60s)
- Sufficient reasoning quality for structured JSON output and narrative generation
- Cost-efficient for production batch jobs

**API key handling:** Read from `GEMINI_API_KEY` environment variable (or `.env` file via `python-dotenv`). Never hardcoded. Never committed to version control.

---

### 2.5 Typed Data Contracts via Python Dataclasses

**Decision:** Inter-agent communication uses typed Python `dataclass` objects (`DataProfile`, `AnalysisResult`) rather than raw dictionaries or DataFrames.

**Justification:**

- **Explicit contracts** — every field is named and typed; no implicit knowledge of dict keys required
- **IDE support** — autocompletion and static analysis work out of the box
- **Fail-fast** — missing or incorrectly typed fields raise errors at construction time, not silently at report rendering
- **No overhead** — dataclasses are pure Python with zero runtime cost

**Key data contracts:**

```python
@dataclass
class DataProfile:
    analysis_goal: str           # propagated to all downstream agents
    target_field: str            # LLM-identified target variable
    fraud_signal_fields: list    # LLM-identified high-signal columns
    prescribed_analyses: list    # what Agent 2 must execute and why

@dataclass
class AnalysisResult:
    llm_interpretation: str        # Gemini's narrative of the findings
    llm_actionable_insights: list  # 3 data-backed business recommendations
    feature_signals: list          # top correlated features with LLM explanations
```

---

### 2.6 Graceful Fallback — No Hard Dependency on LLM Availability

**Decision:** Every LLM call has a fallback path. If `GEMINI_API_KEY` is absent or the API call fails, each agent continues with pandas-only results.

**Justification:**
- **Resilience** — the pipeline never crashes due to an API error
- **Testability** — the pipeline can be tested end-to-end without an API key
- **Separation of concerns** — compute correctness (pandas) is independent of narrative quality (LLM)

Fallback is logged as `[WARNING]` and the report footer records `source: fallback` vs. `source: llm`.

---

### 2.7 CLI Entry Point with Configurable Goal

**Decision:** `main.py` exposes `--input`, `--output`, and `--goal` arguments via `argparse`.

**Justification:** The `--goal` argument is what makes the pipeline dataset-agnostic. It allows the same code to be applied to different analytical problems by changing a single command-line parameter. Follows the Unix principle of configurable tools.

---

## 3. Anomaly Detection Strategy

### 3.1 Statistical Outlier Detection: IQR Method on Transaction Amount

**Method** (executed by `AnalysisAgent` using pandas):

```
Q1 = Amount.quantile(0.25)  →  €5.60
Q3 = Amount.quantile(0.75)  →  €77.51
IQR = Q3 - Q1               →  €71.91
upper_bound = Q3 + 3 × IQR  →  €293.24

flagged = transactions where Amount > upper_bound
```

**Results on the dataset:** 20 high-value outlier transactions flagged above the €293.24 threshold (max observed: €25,691.16).

---

### 3.2 Justification for IQR over Alternatives

| Method | Considered | Reason Not Chosen |
|--------|-----------|-------------------|
| **Z-score** | Yes | Assumes normal distribution; transaction amounts are heavily right-skewed — Z-score would under-flag outliers in the tail |
| **Isolation Forest** | Yes | Requires ML model fitting; increases complexity without proportional gain for a rule-based detection step |
| **DBSCAN clustering** | Yes | High computational cost on 284K rows; difficult to explain to non-technical stakeholders |
| **Fixed threshold** | Yes | Arbitrary; not generalisable to datasets with different amount ranges |
| **IQR (chosen)** ✅ | — | Distribution-agnostic; robust to skew; no model training; transparent and explainable; standard in financial analytics |

**Why 3× IQR instead of 1.5× IQR:**

The standard 1.5× IQR (Tukey fences) would flag ~7% of all transactions as outliers in a right-skewed distribution like transaction amounts — excessive false positives on legitimate high-value purchases. The 3× multiplier produces a more precise set of genuinely extreme values.

---

### 3.3 LLM-Interpreted Fraud Signal Detection

The `AnalysisAgent` computes **Pearson correlation** of each feature (V1–V28) with the `Class` label using pandas, then passes the results to Gemini for contextual interpretation.

**What pandas computes:**

| Feature | Correlation |
|---------|-------------|
| V17 | −0.3135 |
| V14 | −0.2934 |
| V12 | −0.2507 |
| V10 | −0.2070 |

**What Gemini interprets (from the generated report):**

> *"Primary predictive vector; strong negative deviations indicate severe fraud risk. Prioritize V17 and V14 as core split features in XGBoost/LightGBM production pipelines."*

This interpretation is dynamic — it changes if the dataset changes, because it is generated by the LLM from the actual correlation values, not pre-written.

---

### 3.4 Temporal Anomaly Detection

The pipeline bins the `Time` column into 1-hour windows and computes the fraud rate per window. Agent 1 prescribes this analysis with an explicit `goal_link`:

> *"Identifies time windows where fraud risk is elevated — directly supports the goal of detecting fraudulent transactions."*

**Top findings:**

| Hour Bin | Fraud Rate | Baseline Multiplier | Threat Level |
|----------|-----------|---------------------|-------------|
| 26 | **1.556%** | 9.3× | CRITICAL |
| 28 | **1.515%** | 9.1× | CRITICAL |
| 2  | **1.335%** | 8.0× | CRITICAL |

---

### 3.5 Why Standard Accuracy Was Not Used

With only 0.167% of transactions being fraudulent, a trivial classifier that predicts "legitimate" for every transaction achieves **99.83% accuracy** — yet catches zero fraud cases. Gemini explicitly flags this in the generated executive report and recommends **AUPRC (Area Under the Precision-Recall Curve)** and SMOTE-based class balancing for any downstream ML work.

---

## 4. Limitations and Future Work

| Limitation | Suggested Improvement |
|------------|-----------------------|
| IQR detects only Amount outliers; V-feature outliers are not flagged directly | Apply Mahalanobis distance or Isolation Forest on the full feature space |
| LLM calls add ~75 seconds to pipeline runtime | Cache `DataProfile` for repeated runs on the same dataset |
| No customer ID in the dataset; customer-level analysis is not possible | Request unmasked data from provider under NDA |
| Pipeline is batch-only; not suitable for real-time fraud prevention | Add a streaming layer (e.g., Kafka + Faust) for online scoring |
| Report is static Markdown; no interactive visualisations | Integrate with a BI tool (e.g., Metabase, Grafana) or add matplotlib chart exports |

---

## 5. References

- Dataset: [Credit Card Fraud Detection — Kaggle / ULB Machine Learning Group](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- IQR method: Tukey, J.W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
- AUPRC recommendation: Davis, J. & Goadrich, M. (2006). *The relationship between Precision-Recall and ROC curves*. ICML.
- Google Gemini SDK: [https://ai.google.dev/gemini-api/docs](https://ai.google.dev/gemini-api/docs)
- Architecture: [design/design.md](./design/design.md)
- Specification: [spec/spec.md](./spec/spec.md)
