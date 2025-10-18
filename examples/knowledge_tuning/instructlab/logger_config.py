import os
import logging
from rich.logging import RichHandler


def setup_logger(name):
    # Set up the logger
    log_level = os.getenv("LOG_LEVEL", "INFO")
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()],
    )
    logger = logging.getLogger(name)
    return logger
