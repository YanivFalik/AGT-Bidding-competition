from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set


@dataclass
class Belief:
    p_high: float
    p_mixed: float
    p_low: float

"""
Main Flow:
Init:
    - calculate all priors as value group size / number of items
After each round, given item i with private valuation v and paid price p
    - calculate P(T_i=t | v,p) for every t (using priors for all t)
    - calculate E[remaining items in t] for every t
    - update priors of all t (using P(T_j=t | v,p) of all seen items j) 
"""
class ItemBeliefs:
    # Fixed totals (counts)
    TOTAL_HIGH = 6
    TOTAL_MIXED = 10
    TOTAL_LOW = 4
    TOTAL_ITEMS = TOTAL_HIGH + TOTAL_MIXED + TOTAL_LOW

    LOW_RANGE = [1, 10]
    HIGH_RANGE = [10, 20]
    MIXED_RANGE = [1, 20]

    def __init__(self, valuation_vector: Dict[str, float]):
        self.valuation_vector = dict(valuation_vector)

        # Current global priors (start from totals / 20)
        self.prior_high = self.TOTAL_HIGH / self.TOTAL_ITEMS
        self.prior_mixed = self.TOTAL_MIXED / self.TOTAL_ITEMS
        self.prior_low = self.TOTAL_LOW / self.TOTAL_ITEMS

        self.beliefs: Dict[str, Belief] = {}
        self.seen_items: Set[str] = set()  # items already auctioned / revealed

        # Initial beliefs using initial priors
        for item_id, v in self.valuation_vector.items():
            self.beliefs[item_id] = self._posterior_from_value(float(v))

    def get(self, item_id: str) -> Belief:
        return self.beliefs[item_id]

    """
    Bayes’ rule for continuous observations:
    Given PDF f, item i with value v, value group T (high, mixed, low) we get:
    
    P(T_i=t|v) = P(T_i=t)f(v|T_i=t) / sum_t'(P(T_i=t')f(v|T_i=t'))
    
    Where P(T_i=t) is the global prior updated each round
    """

    def _posterior_from_value(self, v: float) -> Belief:
        # calculate f(v|T=t) for every value group
        f_high = (1.0 / (self.HIGH_RANGE[1] - self.HIGH_RANGE[0])) if (
                    self.HIGH_RANGE[0] <= v <= self.HIGH_RANGE[1]) else 0.0
        f_low = (1.0 / (self.LOW_RANGE[1] - self.LOW_RANGE[0])) if (
                    self.LOW_RANGE[0] <= v <= self.LOW_RANGE[1]) else 0.0
        f_mixed = (1.0 / (self.MIXED_RANGE[1] - self.MIXED_RANGE[0])) if (
                    self.MIXED_RANGE[0] <= v <= self.MIXED_RANGE[1]) else 0.0

        # calculate P(T=t)f(v|T=t)
        w_high = self.prior_high * f_high
        w_mixed = self.prior_mixed * f_mixed
        w_low = self.prior_low * f_low
        w_sum = w_high + w_mixed + w_low

        # create new belief with ( P(T=high|v), P(T=mixed|v), P(T=low|v) )
        return Belief(w_high / w_sum, w_mixed / w_sum, w_low / w_sum)

    """
        Bayes’ rule for continuous observations:
        Given:
         - PDF f
         - second highes bid (paid price) p
         - priors for the item being in all item groups P(T=t|v) 
         - value group T (high, mixed, low)

        We can calculate the probability of the item being in each item group depending on the prior and the price paid: 
        P(T=t|v,p) = P(T=t|v)l(p|T=t) / sum_t'(P(T=t'|v)l(p|T=t'))
        
        Where l is the likelihood of the second highest bid being p for an item of group t  
    """
    def update_with_price(self, item_id: str, price_paid: float) -> None:
        prior_item = self.beliefs[item_id]

        """
        For n i.i.d. samples from Uniform[0,1], the density of the k-th order statistic (k-th item in increasing order) is:
        f_k(y) = n!/((k-1)!(n-k)!)y^(k-1)(1-y)^(n-k)
        Where:
         - y = the value of the density function (which is also the probability to be lower than y because the samples are from Uniform[0,1]).
         - k-1 = items below y
         - n-k = items above y
        
        For:
         - y = (p - min)/(max - min)
         - n = 5 (number of agents)
         - k = 4 (second highest = 4 statistic)
        We get:
        f_4(y) = 20y^3(1-y) / (min - max)
        
        Where the division by (min - max) is to shift our density to the range [min,max] instead of [0,1]
        """
        def second_highest_pdf_uniform(p_val: float, range_min: float, range_max: float) -> float:
            y = (p_val - range_min) / (range_max - range_min)
            return 20 * (y ** 3) * (1.0 - y) * (1.0 / (range_max - range_min))

        like_low = second_highest_pdf_uniform(price_paid, self.LOW_RANGE[0], self.LOW_RANGE[1])
        like_high = second_highest_pdf_uniform(price_paid, self.HIGH_RANGE[0], self.HIGH_RANGE[1])
        like_mixed = second_highest_pdf_uniform(price_paid, self.MIXED_RANGE[0], self.MIXED_RANGE[1])

        w_high = prior_item.p_high * like_high
        w_mixed = prior_item.p_mixed * like_mixed
        w_low = prior_item.p_low * like_low

        w_sum = w_high + w_mixed + w_low

        # Update item beliefs
        self.beliefs[item_id] = Belief(w_high / w_sum, w_mixed / w_sum, w_low / w_sum)

        # Mark item as seen, then update global priors and refresh unseen items
        self.seen_items.add(item_id)
        self._update_global_priors_and_refresh_unseen()

    """
    In order to guess the remaining number of items in a value group t we can calculate the expected remainder as follows:
        E[remaining items in t] = TOTAL_T - E[removed items in t]
    Where:
        E[removed items in t] = sum_{i : removed items}P(T_i = t)
        
    Now we can estimate the probability of a random item i being in value group t as:
        P(T_i=t) = E[remaining items in t]/total remaining items
    """
    def _update_global_priors_and_refresh_unseen(self) -> None:
        if len(self.seen_items) == self.TOTAL_ITEMS:
            return

        removed_from_high = sum(self.beliefs[i].p_high for i in self.seen_items)
        removed_from_mixed = sum(self.beliefs[i].p_mixed for i in self.seen_items)
        removed_from_low = sum(self.beliefs[i].p_low for i in self.seen_items)

        remaining_high = max(self.TOTAL_HIGH - removed_from_high, 0)
        remaining_mixed = max(self.TOTAL_MIXED - removed_from_mixed, 0)
        remaining_low = max(self.TOTAL_LOW - removed_from_low, 0)

        # update priors to be E[items left in t]/items left in auction
        items_left = self.TOTAL_ITEMS - len(self.seen_items)
        self.prior_high = remaining_high / items_left
        self.prior_mixed = remaining_mixed / items_left
        self.prior_low = remaining_low / items_left

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
            "item_id   value   P(High)   P(Mixed)    P(Low)"
        )
        for item_id, v in items:
            b = self.beliefs[item_id]
            lines.append(
                f"{item_id:<9} {v:>5.1f}   "
                f"{fmt.format(b.p_high):>7}   {fmt.format(b.p_mixed):>7}   {fmt.format(b.p_low):>7}"
                + ("   [SEEN]" if item_id in self.seen_items else "")
            )
        lines.append(
            f"\nGlobal priors for unseen items: "
            f"High={self.prior_high:.3f}, MIXED={self.prior_mixed:.3f}, Low={self.prior_low:.3f}"
        )
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.to_str()
