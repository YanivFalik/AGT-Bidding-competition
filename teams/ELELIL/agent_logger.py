import logging
import os
from typing import Dict


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

def beliefs_summary(valuation_vector, beliefs, seen_items, priors, digits: int = 3) -> str:
    fmt = f"{{:.{digits}f}}"
    items = sorted(valuation_vector.items(), key=lambda x: x[1], reverse=True)

    lines = []
    lines.append(
        "Item Beliefs (initial or updated)\n"
        "--------------------------------\n"
        "item_id   value   P(High)   P(Mixed)    P(Low)"
    )
    for item_id, v in items:
        b = beliefs[item_id]
        lines.append(
            f"{item_id:<9} {v:>5.1f}   "
            f"{fmt.format(b.p_high):>7}   {fmt.format(b.p_mixed):>7}   {fmt.format(b.p_low):>7}"
            + ("   [SEEN]" if item_id in seen_items else "")
        )
    lines.append(
        f"\nGlobal priors for unseen items: "
        f"High={priors.p_high:.3f}, MIXED={priors.p_mixed:.3f}, Low={priors.p_low:.3f}"
    )
    return "\n".join(lines)