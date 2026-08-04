"""
Logging setup and utilities using Loguru.
"""

from pathlib import Path
from typing import Optional
from loguru import logger
import sys


def setup_logger(
    name: str = "PixCell",
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_file: bool = True,
    log_to_console: bool = True,
) -> None:
    """
    Setup logger with file and console handlers using Loguru.

    Args:
        name: Logger name (not used with loguru, kept for compatibility)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (if None, uses default)
        log_to_file: Whether to log to file
        log_to_console: Whether to log to console
    """
    # Remove default handler
    logger.remove()

    # Console handler
    if log_to_console:
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=log_level.upper(),
            colorize=True,
        )

    # File handler
    if log_to_file and log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        logger.add(
            log_path,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            level=log_level.upper(),
            rotation="10 MB",  # Rotate when file reaches 10MB
            retention="7 days",  # Keep logs for 7 days
            compression="zip",  # Compress old logs
        )

