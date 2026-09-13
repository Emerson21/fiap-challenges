# tools/logger.py
"""
Centralised logging configuration for the pipeline.
All agents should obtain their logger via get_logger(__name__).
"""
import logging

LOG_FORMAT = "%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Ensure the root logger is configured only once
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with the pipeline's standard format.

    Args:
        name (str): Typically __name__ of the calling module.
        level (int): Logging level (default: INFO).

    Returns:
        logging.Logger: Configured logger instance.

    Usage:
        from tools.logger import get_logger
        log = get_logger(__name__)
        log.info("Agent started")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    return logger
