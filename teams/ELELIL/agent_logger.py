import logging
import os

def _make_agent_logger(name: str = "agent_tmp_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # IMPORTANT: don't pass messages to root/main handlers

    # Add handler only once (avoid duplicate lines if module is reloaded)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_path = os.path.join(os.path.dirname(__file__), "tmp.log")
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

    return logger