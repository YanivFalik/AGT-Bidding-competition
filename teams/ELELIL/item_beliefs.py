from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set


@dataclass
class Belief:
    p_high: float
    p_all: float
    p_low: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.p_high, self.p_all, self.p_low)


class ItemBeliefs:
    # Fixed totals (counts)
    TOTAL_HIGH = 6
    TOTAL_ALL = 10
    TOTAL_LOW = 4
    TOTAL_ITEMS = 20

    def __init__(self, valuation_vector: Dict[str, float]):
        self.valuation_vector = dict(valuation_vector)

        # Current global priors (start from totals / 20)
        self.prior_high = self.TOTAL_HIGH / self.TOTAL_ITEMS
        self.prior_all  = self.TOTAL_ALL  / self.TOTAL_ITEMS
        self.prior_low  = self.TOTAL_LOW  / self.TOTAL_ITEMS

        self.beliefs: Dict[str, Belief] = {}
        self.seen_items: Set[str] = set()  # items already auctioned / revealed

        # Initial beliefs using initial priors
        for item_id, v in self.valuation_vector.items():
            self.beliefs[item_id] = self._posterior_from_value(float(v))

    def get(self, item_id: str) -> Belief:
        return self.beliefs[item_id]

    # --- value-based posterior using CURRENT global priors ---
    def _posterior_from_value(self, v: float) -> Belief:
        # Likelihoods (uniform densities)
        f_high = (1.0 / 10.0) if (10.0 <= v <= 20.0) else 0.0
        f_low  = (1.0 / 9.0)  if (1.0 <= v <= 10.0) else 0.0
        f_all  = (1.0 / 19.0) if (1.0 <= v <= 20.0) else 0.0

        w_high = self.prior_high * f_high
        w_all  = self.prior_all  * f_all
        w_low  = self.prior_low  * f_low

        Z = w_high + w_all + w_low
        if Z <= 0:
            return Belief(0.0, 1.0, 0.0)

        return Belief(w_high / Z, w_all / Z, w_low / Z)

    def update_with_price(self, item_id: str, price_paid: float) -> None:
        """Update belief for THIS item using price, then update global priors and refresh unseen items."""
        if item_id not in self.beliefs:
            return

        prior_item = self.beliefs[item_id]
        p = float(price_paid)

        def second_highest_pdf_uniform(p_val: float, a: float, b: float) -> float:
            if b <= a or p_val < a or p_val > b:
                return 0.0
            y = (p_val - a) / (b - a)
            return 12.0 * (y ** 2) * (1.0 - y) * (1.0 / (b - a))

        like_low  = second_highest_pdf_uniform(p, 1.0, 10.0)
        like_high = second_highest_pdf_uniform(p, 10.0, 20.0)
        like_all  = second_highest_pdf_uniform(p, 1.0, 20.0)

        w_high = prior_item.p_high * like_high
        w_all  = prior_item.p_all  * like_all
        w_low  = prior_item.p_low  * like_low

        Z = w_high + w_all + w_low
        if Z <= 0.0:
            # Still mark as seen (it was auctioned), but don't change its belief
            self.seen_items.add(item_id)
            self._update_global_priors_and_refresh_unseen()
            return

        # Update item belief
        self.beliefs[item_id] = Belief(w_high / Z, w_all / Z, w_low / Z)

        # Mark item as seen, then update global priors and refresh unseen items
        self.seen_items.add(item_id)
        self._update_global_priors_and_refresh_unseen()

    def _update_global_priors_and_refresh_unseen(self) -> None:
        """
        Mean-field / expected-count update:
          expected used counts = sum of posteriors over seen items
          remaining counts = totals - expected used
          new priors for unseen items = remaining / unseen_count
        Then recompute P(T|v) for all unseen items using the NEW priors.
        """
        used_high = sum(self.beliefs[i].p_high for i in self.seen_items)
        used_all  = sum(self.beliefs[i].p_all  for i in self.seen_items)
        used_low  = sum(self.beliefs[i].p_low  for i in self.seen_items)

        rem_high = self.TOTAL_HIGH - used_high
        rem_all  = self.TOTAL_ALL  - used_all
        rem_low  = self.TOTAL_LOW  - used_low

        unseen = self.TOTAL_ITEMS - len(self.seen_items)
        if unseen <= 0:
            return

        # Clamp to avoid negative priors due to numerical drift / model mismatch
        rem_high = max(0.0, rem_high)
        rem_all  = max(0.0, rem_all)
        rem_low  = max(0.0, rem_low)

        total_rem = rem_high + rem_all + rem_low
        if total_rem <= 0:
            return

        # New global priors for an unseen item
        self.prior_high = rem_high / unseen
        self.prior_all  = rem_all  / unseen
        self.prior_low  = rem_low  / unseen

        # Recompute beliefs for unseen items from value only, using updated priors
        for item_id, v in self.valuation_vector.items():
            if item_id in self.seen_items:
                continue
            self.beliefs[item_id] = self._posterior_from_value(float(v))

    # NOTE: __str__ should NOT take digits param; Python expects __str__(self) only.
    def to_str(self, digits: int = 3) -> str:
        fmt = f"{{:.{digits}f}}"
        items = sorted(self.valuation_vector.items(), key=lambda x: x[1], reverse=True)

        lines = []
        lines.append(
            "Item Beliefs (initial or updated)\n"
            "--------------------------------\n"
            "item_id   value   P(High)   P(All)    P(Low)"
        )
        for item_id, v in items:
            b = self.beliefs[item_id]
            lines.append(
                f"{item_id:<9} {v:>5.1f}   "
                f"{fmt.format(b.p_high):>7}   {fmt.format(b.p_all):>7}   {fmt.format(b.p_low):>7}"
                + ("   [SEEN]" if item_id in self.seen_items else "")
            )
        lines.append(
            f"\nGlobal priors for unseen items: "
            f"High={self.prior_high:.3f}, All={self.prior_all:.3f}, Low={self.prior_low:.3f}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_str()
