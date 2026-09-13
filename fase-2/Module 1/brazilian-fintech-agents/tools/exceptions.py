# tools/exceptions.py
"""
Custom exception classes for the Brazilian Fintech Agents pipeline.
Raised by IngestionAgent when input data fails validation.
"""


class SchemaValidationError(Exception):
    """
    Raised when the CSV file is missing one or more required columns.

    Example:
        raise SchemaValidationError("Missing columns: ['V14', 'Class']")
    """
    pass


class DataQualityError(Exception):
    """
    Raised when data quality thresholds are exceeded (e.g., >20% nulls
    in any column).

    Example:
        raise DataQualityError("Column 'Amount' has 35.2% null values (threshold: 20%)")
    """
    pass
