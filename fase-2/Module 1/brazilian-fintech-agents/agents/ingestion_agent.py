# agents/ingestion_agent.py
"""
Agent 1 — Ingestion & Validation

Responsible for:
  - Loading the raw CSV file
  - Validating schema (required columns)
  - Checking and imputing null values
  - Removing duplicate rows
  - Coercing data types
  - Producing a DataSummary for downstream agents
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple

from models.data_summary import DataSummary, DateRange, AmountStats
from tools.exceptions import SchemaValidationError, DataQualityError
from tools.logger import get_logger

log = get_logger(__name__)


class IngestionAgent:
    """Loads, validates, cleans and summarises the raw transaction CSV."""

    VERSION = "0.1.0"

    REQUIRED_COLUMNS = (
        ["Time", "Amount", "Class"] + [f"V{i}" for i in range(1, 29)]
    )
    NULL_THRESHOLD_PCT = 0.20  # raise DataQualityError if any col exceeds this

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, csv_path: str) -> Tuple[pd.DataFrame, DataSummary]:
        """
        Execute the full ingestion pipeline.

        Args:
            csv_path (str): Path to creditcard.csv.

        Returns:
            Tuple[pd.DataFrame, DataSummary]:
                - Cleaned dataframe ready for EDA
                - DataSummary with metadata about the loaded data

        Raises:
            FileNotFoundError: if csv_path does not exist.
            SchemaValidationError: if required columns are missing.
            DataQualityError: if null threshold is exceeded in any column.
        """
        log.info("IngestionAgent v%s started — input: %s", self.VERSION, csv_path)

        df = self._load(csv_path)
        self._validate_schema(df)
        missing = self._check_nulls(df)
        df = self._impute_nulls(df)
        df, dupes_removed = self._remove_duplicates(df)
        df = self._coerce_types(df)
        summary = self._compute_summary(df, missing, dupes_removed)

        log.info(
            "IngestionAgent complete — shape: %s | duplicates removed: %d",
            df.shape,
            dupes_removed,
        )
        return df, summary

    # ------------------------------------------------------------------
    # Private steps
    # ------------------------------------------------------------------

    def _load(self, csv_path: str) -> pd.DataFrame:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        log.info("Loading CSV from %s ...", csv_path)
        df = pd.read_csv(path)
        log.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def _validate_schema(self, df: pd.DataFrame) -> None:
        missing_cols = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing_cols:
            raise SchemaValidationError(
                f"CSV is missing required columns: {missing_cols}"
            )
        log.info("Schema validation passed — all %d required columns present", len(self.REQUIRED_COLUMNS))

    def _check_nulls(self, df: pd.DataFrame) -> dict:
        """Return null counts per column; raise if any column exceeds threshold."""
        null_counts = df.isnull().sum().to_dict()
        n_rows = len(df)
        violations = {
            col: count
            for col, count in null_counts.items()
            if count / n_rows > self.NULL_THRESHOLD_PCT
        }
        if violations:
            details = ", ".join(
                f"'{col}': {count/n_rows:.1%}" for col, count in violations.items()
            )
            raise DataQualityError(
                f"Null threshold ({self.NULL_THRESHOLD_PCT:.0%}) exceeded — {details}"
            )
        total_nulls = sum(null_counts.values())
        log.info("Null check passed — total nulls found: %d", total_nulls)
        return null_counts

    def _impute_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill numeric nulls with the column median."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                log.info("Imputed nulls in '%s' with median=%.4f", col, median_val)
        return df

    def _remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Drop exact duplicate rows and return updated df + count removed."""
        before = len(df)
        df = df.drop_duplicates()
        removed = before - len(df)
        if removed:
            log.warning("Removed %d duplicate rows", removed)
        else:
            log.info("No duplicate rows found")
        return df, removed

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Class is int64 and Amount / Time are float64."""
        df["Class"] = df["Class"].astype("int64")
        df["Amount"] = df["Amount"].astype("float64")
        df["Time"] = df["Time"].astype("float64")
        log.info("Type coercion complete")
        return df

    def _compute_summary(
        self,
        df: pd.DataFrame,
        missing_values: dict,
        dupes_removed: int,
    ) -> DataSummary:
        """Build and return the DataSummary dataclass."""
        min_t = df["Time"].min()
        max_t = df["Time"].max()

        return DataSummary(
            shape=df.shape,
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            missing_values=missing_values,
            duplicates_removed=dupes_removed,
            date_range=DateRange(
                min_time_sec=float(min_t),
                max_time_sec=float(max_t),
                span_hours=float((max_t - min_t) / 3600),
            ),
            class_distribution=df["Class"].value_counts().to_dict(),
            amount_stats=AmountStats(
                min=float(df["Amount"].min()),
                max=float(df["Amount"].max()),
                mean=float(df["Amount"].mean()),
                std=float(df["Amount"].std()),
                median=float(df["Amount"].median()),
            ),
        )
