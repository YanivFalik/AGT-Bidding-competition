""" ============================================================================================================
================================= BELIEF CALCULATIONS ==========================================================
============================================================================================================ """
from dataclasses import dataclass
from typing import Dict, Tuple

TOTAL_HIGH = 6
TOTAL_MIXED = 10
TOTAL_LOW = 4
TOTAL_ITEMS = TOTAL_HIGH + TOTAL_MIXED + TOTAL_LOW

VALUE_RANGE_LOW = [1,10]
VALUE_RANGE_MIXED = [1,20]
VALUE_RANGE_HIGH = [10,20]

@dataclass
class Belief:
    p_high: float
    p_mixed: float
    p_low: float

def get_posteriors_from_values(items: Dict[str, float], priors: Belief) -> Dict[str, Belief]:
    return {item_id: posterior_from_value(float(v), priors) for item_id, v in items.items()}

def posterior_from_value(v: float, priors: Belief, possible_highs: list[str] = None, possible_lows: list[str] = None) -> Belief:
    # if the number of possible low items is exactly the size of the low item group,
    # an item with value of less than 10 is surely low
    if v >= VALUE_RANGE_HIGH[0]:
        if possible_highs and len(possible_highs) == TOTAL_HIGH:
            return Belief(1, 0, 0)

    if v <= VALUE_RANGE_LOW[1]:
        if possible_lows and len(possible_lows) == TOTAL_LOW:
            return Belief(0, 0, 1)

    # calculate f(v|T=t) for every value group
    f_high = (1.0 / (VALUE_RANGE_HIGH[1] - VALUE_RANGE_HIGH[0])) if (
            VALUE_RANGE_HIGH[0] <= v <= VALUE_RANGE_HIGH[1]) else 0.0
    f_low = (1.0 / (VALUE_RANGE_LOW[1] - VALUE_RANGE_LOW[0])) if (
            VALUE_RANGE_LOW[0] <= v <= VALUE_RANGE_LOW[1]) else 0.0
    f_mixed = (1.0 / (VALUE_RANGE_MIXED[1] - VALUE_RANGE_MIXED[0])) if (
            VALUE_RANGE_MIXED[0] <= v <= VALUE_RANGE_MIXED[1]) else 0.0

    # calculate P(T=t)f(v|T=t)
    w_high = priors.p_high * f_high
    w_mixed = priors.p_mixed * f_mixed
    w_low = priors.p_low * f_low
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

MIXED_ENSURANCE_THRESHOLD = 0.1
def get_posterior_with_price(item_id: str, value: float, price_paid: float, posterior_without_price: Belief) -> Belief:
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
        if p_val < range_min or p_val > range_max:
            return 0.0
        y = (p_val - range_min) / (range_max - range_min)
        return 20 * (y ** 3) * (1.0 - y) * (1.0 / (range_max - range_min))

    if posterior_without_price.p_low == 1.0 and price_paid < VALUE_RANGE_LOW[1] or posterior_without_price.p_high == 1.0 and price_paid > \
            VALUE_RANGE_HIGH[0]:
        return posterior_without_price

    like_low = second_highest_pdf_uniform(price_paid, VALUE_RANGE_LOW[0], VALUE_RANGE_LOW[1])
    like_high = second_highest_pdf_uniform(price_paid, VALUE_RANGE_HIGH[0], VALUE_RANGE_HIGH[1])
    like_mixed = second_highest_pdf_uniform(price_paid, VALUE_RANGE_MIXED[0], VALUE_RANGE_MIXED[1])

    w_high = posterior_without_price.p_high * like_high
    w_mixed = posterior_without_price.p_mixed * like_mixed
    w_low = posterior_without_price.p_low * like_low

    w_sum = w_high + w_mixed + w_low

    min_val = min(price_paid, value)
    max_val = max(price_paid, value)
    if min_val <= (VALUE_RANGE_LOW[1] - MIXED_ENSURANCE_THRESHOLD) and max_val >= (
            VALUE_RANGE_HIGH[0] + MIXED_ENSURANCE_THRESHOLD):
        return Belief(0, 1, 0)

    return Belief(w_high / w_sum, w_mixed / w_sum, w_low / w_sum)

def get_group_possible_candidates(valuation_vector: Dict[str, float], beliefs: Dict[str, Belief]) -> Tuple[list[str], list[str]]:
    possible_highs = [item_id for item_id, v in valuation_vector.items() if
                           v >= VALUE_RANGE_HIGH[0]
                           and (item_id not in beliefs or beliefs[item_id].p_mixed < 1)]
    possible_lows = [item_id for item_id, v in valuation_vector.items() if
                           v <= VALUE_RANGE_LOW[1]
                          and (item_id not in beliefs or beliefs[item_id].p_mixed < 1)]
    return possible_highs, possible_lows

def get_expected_remainders_from_seen(beliefs: Dict[str, Belief], seen_items_and_prices: set[str]) -> Belief:
    expected_used_high = sum(beliefs[i].p_high for i in seen_items_and_prices)
    expected_used_mixed = sum(beliefs[i].p_mixed for i in seen_items_and_prices)
    expected_used_low = sum(beliefs[i].p_low for i in seen_items_and_prices)

    # Keep nonnegative (should already be, unless upstream logic forces too many)
    expected_high_remainder = max(0.0, TOTAL_HIGH - expected_used_high)
    expected_mixed_remainder = max(0.0, TOTAL_MIXED - expected_used_mixed)
    expected_low_remainder = max(0.0, TOTAL_LOW - expected_used_low)
    return Belief(expected_high_remainder, expected_mixed_remainder, expected_low_remainder)

""" ============================================================================================================
=============================== END BELIEF CALCULATIONS ========================================================
============================================================================================================ """