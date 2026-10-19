import logging
import os
from datetime import datetime


def setup_logger(
    name: str = "xai_logger",
    base_dir: str = None,
    level=logging.INFO
):
    """
    Tworzy logger:
    - konsola
    - plik: xai_logs/<timestamp>/xai.log
    """

    if base_dir is None:
        base_dir = os.getcwd()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    log_dir = os.path.join(
        base_dir,
        "shared",
        "files",
        "xai_logs",
        timestamp
    )
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "xai.log")

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info("Logger initialized")
    logger.info(f"Log directory: {log_dir}")

    return logger
