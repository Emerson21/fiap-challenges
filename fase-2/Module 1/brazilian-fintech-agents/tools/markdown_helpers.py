# tools/markdown_helpers.py
"""
Utility functions for building Markdown content programmatically.
Used exclusively by ReportWriterAgent to assemble report sections.
"""
from typing import List


def make_section(title: str, level: int = 2) -> str:
    """
    Return a Markdown section header.

    Args:
        title (str): Section title text.
        level (int): Heading level 1–6 (default: 2).

    Returns:
        str: e.g. "## My Section\\n"
    """
    prefix = "#" * level
    return f"{prefix} {title}\n"


def make_table(headers: List[str], rows: List[List[str]]) -> str:
    """
    Build a GitHub-flavoured Markdown table.

    Args:
        headers (List[str]): Column header names.
        rows (List[List[str]]): Table data rows; each inner list must
                                have the same length as headers.

    Returns:
        str: Fully formatted Markdown table string.

    Example:
        make_table(["Name", "Value"], [["fraud_rate", "0.17%"]])
        # | Name       | Value |
        # |------------|-------|
        # | fraud_rate | 0.17% |
    """
    col_widths = [
        max(len(h), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt_row(cells: List[str]) -> str:
        padded = [str(c).ljust(col_widths[i]) for i, c in enumerate(cells)]
        return "| " + " | ".join(padded) + " |"

    separator = "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt_row(headers), separator] + [fmt_row(row) for row in rows]
    return "\n".join(lines) + "\n"


def make_insight(number: int, title: str, finding: str, action: str) -> str:
    """
    Format a single actionable insight block.

    Args:
        number (int): Insight sequence number (1, 2, 3).
        title (str): Short insight title.
        finding (str): What the data shows.
        action (str): Recommended business action.

    Returns:
        str: Formatted Markdown block for one insight.
    """
    return (
        f"### Insight {number}: {title}\n\n"
        f"**Finding:** {finding}\n\n"
        f"**Recommended Action:** {action}\n"
    )
