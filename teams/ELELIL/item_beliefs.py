from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Set

from teams.ELELIL.belief_functional import Belief, get_posteriors_from_values, get_posterior_with_price, \
    get_group_possible_candidates, get_expected_remainders_from_seen

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

    MIXED_ENSURANCE_THRESHOLD = 1

    def __init__(self, valuation_vector: Dict[str, float]):
        self.valuation_vector = dict(valuation_vector)

        # Current global priors (start from totals / 20)
        self.prior_high = self.TOTAL_HIGH / self.TOTAL_ITEMS
        self.prior_mixed = self.TOTAL_MIXED / self.TOTAL_ITEMS
        self.prior_low = self.TOTAL_LOW / self.TOTAL_ITEMS

        self.beliefs: Dict[str, Belief] = {}
        self.seen_items: Set[str] = set()  # items already auctioned / revealed

        self.expected_high_remainder = self.TOTAL_HIGH
        self.expected_mixed_remainder = self.TOTAL_MIXED
        self.expected_low_remainder = self.TOTAL_LOW


        self.possible_highs, self.possible_lows = get_group_possible_candidates(valuation_vector, {})
        self.beliefs = get_posteriors_from_values(valuation_vector, Belief(self.prior_high, self.prior_mixed, self.prior_low))


    def get(self, item_id: str) -> Belief:
        return self.beliefs[item_id]

    def update_according_to_price(self, item_id: str, price_paid: float):
        # For price p of item i with value v:
        # calculate P(T_i=t | v,p) for every t in [LOW,MIXED,HIGH]
        self.beliefs[item_id] = get_posterior_with_price(item_id, self.valuation_vector[item_id], price_paid, self.beliefs[item_id])
        self.seen_items.add(item_id)

        # SPECIAL CASE HANDLING: in case of P(T_i=MIXED)=1, remove i from the list of possible LOW/HIGH items
        self.possible_highs, self.possible_lows = get_group_possible_candidates(self.valuation_vector, {})

        # Calculate E[items remaining in group t] for every t in [LOW,MIXED,HIGH]
        remainders = get_expected_remainders_from_seen(self.beliefs, self.seen_items)
        self.expected_high_remainder = remainders.p_high
        self.expected_mixed_remainder = remainders.p_mixed
        self.expected_low_remainder = remainders.p_low

        # Update P(T_i=t) to be E[items remaining in t]/E[items remaining] for every t in [LOW,MIXED,HIGH]
        self._update_global_priors()

        # Update P(T_j | v) for all remaining items
        self._update_posteriors_of_unseens()

    # def _recompute_remainders_from_seen(self):
    #     expected_used_high = sum(self.beliefs[i].p_high for i in self.seen_items)
    #     expected_used_mixed = sum(self.beliefs[i].p_mixed for i in self.seen_items)
    #     expected_used_low = sum(self.beliefs[i].p_low for i in self.seen_items)
    #
    #     # Keep nonnegative (should already be, unless upstream logic forces too many)
    #     self.expected_high_remainder = max(0.0, self.TOTAL_HIGH - expected_used_high)
    #     self.expected_mixed_remainder = max(0.0, self.TOTAL_MIXED - expected_used_mixed)
    #     self.expected_low_remainder = max(0.0, self.TOTAL_LOW - expected_used_low)


    def _update_posteriors_of_unseens(self):
        unseen = {k:v for k,v in self.valuation_vector.items() if k not in self.seen_items}
        posteriors = get_posteriors_from_values(unseen, Belief(self.prior_high, self.prior_mixed, self.prior_low))

        # normalize posteriors such that sum_i(P(T_i=t)) = E[remaining items in t]
        normalized_posteriors = self._normalize_by_group_remainders(posteriors)
        for item_id, belief in normalized_posteriors.items():
            self.beliefs[item_id] = belief

    # def _create_posteriors_from_values(self, items: Dict[str, float]):
    #     return {item_id : self._posterior_from_value(float(v)) for item_id,v in items.items()}


    """
    Update all posteriors such that sum_i(P(T_i=t)) = E[items remaining it t]
    This is done by calculating factors 
    """
    def _normalize_by_group_remainders(self, posteriors: Dict[str, Belief]):
        cumulative_high_prob = sum(belief.p_high for belief in posteriors.values())
        cumulative_mixed_prob = sum(belief.p_mixed for belief in posteriors.values())
        cumulative_low_prob = sum(belief.p_low for belief in posteriors.values())

        high_factor = self.expected_high_remainder / cumulative_high_prob if cumulative_high_prob > 0 else 1.0
        mixed_factor = self.expected_mixed_remainder / cumulative_mixed_prob if cumulative_mixed_prob > 0 else 1.0
        low_factor = self.expected_low_remainder / cumulative_low_prob if cumulative_low_prob > 0 else 1.0

        normalized_posteriors = {}
        for item_id, belief in posteriors.items():
            factored_high = belief.p_high * high_factor
            factored_low = belief.p_low * low_factor
            factored_mixed = belief.p_mixed * mixed_factor
            total_factor = factored_high + factored_low + factored_mixed
            if total_factor == 0:
                normalized_posteriors[item_id] = belief
                continue

            normalized_posteriors[item_id] = Belief(
                factored_high / total_factor,
                factored_mixed / total_factor,
                factored_low / total_factor
            )
        return normalized_posteriors

    # def _update_group_possible_candidates(self):
    #     self.possible_highs = [item_id for item_id, v in self.valuation_vector.items() if
    #                            v >= self.HIGH_RANGE[0]
    #                            and (item_id not in self.beliefs or self.beliefs[item_id].p_mixed < 1)]
    #     self.possible_lows = [item_id for item_id, v in self.valuation_vector.items() if
    #                            v <= self.LOW_RANGE[1]
    #                           and (item_id not in self.beliefs or self.beliefs[item_id].p_mixed < 1)]

    """
    Bayes’ rule for continuous observations:
    Given PDF f, item i with value v, value group T (high, mixed, low) we get:
    
    P(T_i=t|v) = P(T_i=t)f(v|T_i=t) / sum_t'(P(T_i=t')f(v|T_i=t'))
    
    Where P(T_i=t) is the global prior updated each round
    """
    def _posterior_from_value(self, v: float) -> Belief:
        # if the number of possible low items is exactly the size of the low item group,
        # an item with value of less than 10 is surely low
        if v >= self.HIGH_RANGE[0]:
            if len(self.possible_highs) == self.TOTAL_HIGH:
                return Belief(1, 0, 0)

        if v <= self.LOW_RANGE[1]:
            if len(self.possible_lows) == self.TOTAL_LOW:
                return Belief(0, 0, 1)

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

    # """
    #     Bayes’ rule for continuous observations:
    #     Given:
    #      - PDF f
    #      - second highes bid (paid price) p
    #      - priors for the item being in all item groups P(T=t|v)
    #      - value group T (high, mixed, low)
    #
    #     We can calculate the probability of the item being in each item group depending on the prior and the price paid:
    #     P(T=t|v,p) = P(T=t|v)l(p|T=t) / sum_t'(P(T=t'|v)l(p|T=t'))
    #
    #     Where l is the likelihood of the second highest bid being p for an item of group t
    # """
    # def _update_posterior_with_price(self, item_id: str, price_paid: float) -> None:
    #     """
    #     For n i.i.d. samples from Uniform[0,1], the density of the k-th order statistic (k-th item in increasing order) is:
    #     f_k(y) = n!/((k-1)!(n-k)!)y^(k-1)(1-y)^(n-k)
    #     Where:
    #      - y = the value of the density function (which is also the probability to be lower than y because the samples are from Uniform[0,1]).
    #      - k-1 = items below y
    #      - n-k = items above y
    #
    #     For:
    #      - y = (p - min)/(max - min)
    #      - n = 5 (number of agents)
    #      - k = 4 (second highest = 4 statistic)
    #     We get:
    #     f_4(y) = 20y^3(1-y) / (min - max)
    #
    #     Where the division by (min - max) is to shift our density to the range [min,max] instead of [0,1]
    #     """
    #     def second_highest_pdf_uniform(p_val: float, range_min: float, range_max: float) -> float:
    #         if p_val < range_min or p_val > range_max:
    #             return 0.0
    #         y = (p_val - range_min) / (range_max - range_min)
    #         return 20 * (y ** 3) * (1.0 - y) * (1.0 / (range_max - range_min))
    #
    #     prior_item = self.beliefs[item_id]
    #     if prior_item.p_low == 1.0 and price_paid < self.LOW_RANGE[1] or prior_item.p_high == 1.0 and price_paid > self.HIGH_RANGE[0]:
    #         return
    #
    #     like_low = second_highest_pdf_uniform(price_paid, self.LOW_RANGE[0], self.LOW_RANGE[1])
    #     like_high = second_highest_pdf_uniform(price_paid, self.HIGH_RANGE[0], self.HIGH_RANGE[1])
    #     like_mixed = second_highest_pdf_uniform(price_paid, self.MIXED_RANGE[0], self.MIXED_RANGE[1])
    #
    #     w_high = prior_item.p_high * like_high
    #     w_mixed = prior_item.p_mixed * like_mixed
    #     w_low = prior_item.p_low * like_low
    #
    #     w_sum = w_high + w_mixed + w_low
    #
    #     # Update item beliefs
    #     belief: Belief
    #     value = self.valuation_vector[item_id]
    #
    #     min_val = min(price_paid, value)
    #     max_val = max(price_paid, value)
    #     if min_val <= (self.LOW_RANGE[1] - self.MIXED_ENSURANCE_THRESHOLD) and  max_val >= (self.HIGH_RANGE[0] + self.MIXED_ENSURANCE_THRESHOLD):
    #         belief = Belief(0, 1, 0)
    #     else:
    #         belief = Belief(w_high / w_sum, w_mixed / w_sum, w_low / w_sum)
    #
    #     self.beliefs[item_id] = belief


    """
    In order to guess the remaining number of items in a value group t we can calculate the expected remainder as follows:
        E[remaining items in t] = TOTAL_T - E[removed items in t]
    Where:
        E[removed items in t] = sum_{i : removed items}P(T_i = t)
        
    Now we can estimate the probability of a random item i being in value group t as:
        P(T_i=t) = E[remaining items in t]/total remaining items
    """
    def _update_global_priors(self) -> None:
        expected_items_left = self.expected_high_remainder + self.expected_low_remainder + self.expected_mixed_remainder
        if expected_items_left <= 0:
            self.prior_high = 0.0
            self.prior_mixed = 0.0
            self.prior_low = 0.0
            return

        self.prior_high = self.expected_high_remainder / expected_items_left
        self.prior_mixed = self.expected_mixed_remainder / expected_items_left
        self.prior_low = self.expected_low_remainder / expected_items_left

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
