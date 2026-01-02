# - Phase(Round number) -> { Early, Mid, Late }
# - RelativeBudget(Budget, competitor budgets) -> { Above avg, Below avg, Avg }
# - BudgetRank(Budget, competitor budgets) -> [1,5]
# - DollarUtilization(item_value, item_posteriors)* = E[Utility]/E[Price Paid] -> { <=Threshold, >Threshold }
# - RemainingValueProportion(seen_items, item_valuations) = sum(value of remaining items)/sum(value of all items) -> { 0-1/3, 1/3-2/3, 2/3-1 }
# - ExpectedUtilityToRoundProportion** -> { <<1, =1, >>1 }
# - OpponentModeling*** -> { Aggressive, Truthful, Conservative }
# - SuccessRate(seen_items, current_utility) -> { HIGH, MID, LOW }
# - ExpectedUtilityRank(current_utility, seen_items, paid_prices)**** -> [1,5]

from dataclasses import dataclass
from enum import Enum, auto
from typing import Dict, Callable, Tuple

from teams.ELELIL.item_beliefs import Belief

TOTAL_ROUNDS = 15

high_order_statistics = [11+2/3, 13+1/3, 15, 16+2/3, 18+1/3]
mixed_order_statistics = [4+1/6, 7+1/3, 10.5, 13+2/3, 16+5/6]
low_order_statistics = [2.5, 4, 5.5, 7, 8.5]

class RelativeBudgetSignal(Enum):
    BELOW_AVG = auto()
    AVG = auto()
    ABOVE_AVG = auto()

dollar_utilization_threshold = 0.4
class DollarUtilizationSignal(Enum):
    GEQ_THRESHOLD = auto()
    LESS_THRESHOLD = auto()

value_proportion_thresholds = [1/3,2/3]
class RemainingValueProportionSignal(Enum):
    LOW = auto() # 0-1/3
    MID = auto() # 1/3-2/3
    HIGH = auto() # 2/3-1

expected_utility_to_round_threshold = 0.1
class ExpectedUtilityToRoundProportionSignal(Enum):
    BELOW_ONE = auto()
    CLOSE_TO_ONE = auto()
    ABOVE_ONE = auto()

expected_utility_to_round_tolerance = 0.15
class ExpectedUtilityToRoundProportionSignal(Enum):
    MUCH_LESS_1 = auto()
    ABOUT_1 = auto()
    MUCH_GREATER_1 = auto()


@dataclass
class SignalInput:
    item_id: str
    total_rounds: 15
    round_number: int
    our_budget: float
    competitor_budgets: Dict[str, float]
    valuation_vector: Dict[str,float]
    posterior_vector: Dict[str,Belief]
    seen_items_and_prices: Dict[str,float]
    current_utility: float

relative_budget_tolerance = 0.1

def signal_relative_budget(input: SignalInput) -> RelativeBudgetSignal:
    total_budget = sum(input.competitor_budgets.values())+input.our_budget
    ratio = input.our_budget / (total_budget/5)
    print(input.competitor_budgets, total_budget, input.our_budget, ratio)
    if ratio <= 1 - relative_budget_tolerance:
        return RelativeBudgetSignal.BELOW_AVG
    if ratio >= 1 + relative_budget_tolerance:
        return RelativeBudgetSignal.ABOVE_AVG
    return RelativeBudgetSignal.AVG

# add into consideration the probability of being larger than the 4th order statistic
def signal_dollar_utilization(input: SignalInput) -> DollarUtilizationSignal:
    posteriors = input.posterior_vector[input.item_id]
    expected_fourth_order = (
            posteriors.p_high * high_order_statistics[4]
            + posteriors.p_low * low_order_statistics[4]
            + posteriors.p_mixed * mixed_order_statistics[4]
    )
    expected_utilization = (input.valuation_vector[input.item_id] - expected_fourth_order) / expected_fourth_order
    print(f"for item {input.item_id}, dollar utilization: {expected_utilization}")
    return (
        DollarUtilizationSignal.GEQ_THRESHOLD) \
            if expected_utilization >= dollar_utilization_threshold \
            else DollarUtilizationSignal.LESS_THRESHOLD

def signal_remaining_value_proportion(input: SignalInput) -> RemainingValueProportionSignal:
    total_value = sum(input.valuation_vector.values())
    left_value = sum([value for key,value in input.valuation_vector.items() if key not in input.seen_items_and_prices])
    proportion = left_value / total_value
    if proportion <= value_proportion_thresholds[0]:
        return RemainingValueProportionSignal.LOW
    elif proportion <= value_proportion_thresholds[1]:
        return RemainingValueProportionSignal.MID
    return RemainingValueProportionSignal.HIGH

# ExpectedUtilityToRoundProportion = (current_utility / (current_utility + sum(expected utility of remaining items))) / (current_round / total_rounds)
def signal_expected_utility_to_round_proportion(input: SignalInput) -> ExpectedUtilityToRoundProportionSignal:

    def get_expected_utility(value: float, belief: Belief) -> float:
        expected_fourth_order = belief.p_low * low_order_statistics[4] + belief.p_mixed * mixed_order_statistics[4] + belief.p_high * high_order_statistics[4]
        return max(value - expected_fourth_order, 0.0)

    expected_utility_out_of_remaining = [get_expected_utility(value, input.posterior_vector[item_id]) \
                                         for item_id,value in input.valuation_vector.items() \
                                         if item_id not in input.seen_items_and_prices]

    utility_out_of_expected_remaining = input.current_utility / (input.current_utility + sum(expected_utility_out_of_remaining))
    round_out_of_total = input.round_number / TOTAL_ROUNDS

    utility_to_round_proportion = utility_out_of_expected_remaining / round_out_of_total

    if utility_to_round_proportion < (1 - expected_utility_to_round_threshold):
        return ExpectedUtilityToRoundProportionSignal.BELOW_ONE
    elif utility_to_round_proportion <= (1 + expected_utility_to_round_threshold):
        return ExpectedUtilityToRoundProportionSignal.CLOSE_TO_ONE
    return ExpectedUtilityToRoundProportionSignal.ABOVE_ONE



def expected_price_paid_4th_order(input: SignalInput, item_id: str) -> float:
    post = input.posterior_vector[item_id]
    return (
        post.p_high * high_order_statistics[4]
        + post.p_low * low_order_statistics[4]
        + post.p_mixed * mixed_order_statistics[4]
    )

def expected_utility_of_item(input: SignalInput, item_id: str) -> float:
    # expected utility = max(value - expected_price, 0)
    v = input.valuation_vector[item_id]
    p = expected_price_paid_4th_order(input, item_id)
    if p <= 0:
        return 0.0
    return max(v - p, 0.0)

def signal_expected_utility_to_round_proportion(input: SignalInput) -> ExpectedUtilityToRoundProportionSignal:
    # remaining items = items we have valuations+posteriors for, excluding already-seen items
    seen = set(input.seen_items_and_prices.keys())
    candidate_items = set(input.valuation_vector.keys()) & set(input.posterior_vector.keys())
    remaining_items = [iid for iid in candidate_items if iid not in seen]

    remaining_eu = sum(expected_utility_of_item(input, iid) for iid in remaining_items)

    denom = input.current_utility + remaining_eu
    progress_fraction = (input.current_utility / denom) if denom > 0 else 0.0

    # requires input.total_rounds (recommended)
    total_rounds = getattr(input, "total_rounds", None)
    if not total_rounds or total_rounds <= 0:
        raise ValueError("SignalInput must include total_rounds > 0 for ExpectedUtilityToRoundProportion")

    time_fraction = input.round_number / total_rounds
    time_fraction = max(time_fraction, 1e-9)  # avoid division by zero in round 0

    ratio = progress_fraction / time_fraction

    if ratio <= 1.0 - expected_utility_to_round_tolerance:
        return ExpectedUtilityToRoundProportionSignal.MUCH_LESS_1
    if ratio >= 1.0 + expected_utility_to_round_tolerance:
        return ExpectedUtilityToRoundProportionSignal.MUCH_GREATER_1
    return ExpectedUtilityToRoundProportionSignal.ABOUT_1


registered_signals = {
    RelativeBudgetSignal.__name__ : signal_relative_budget,
    DollarUtilizationSignal.__name__ : signal_dollar_utilization,
    RemainingValueProportionSignal.__name__ : signal_remaining_value_proportion,
    ExpectedUtilityToRoundProportionSignal.__name__ : signal_expected_utility_to_round_proportion,
    ExpectedUtilityToRoundProportionSignal.__name__: signal_expected_utility_to_round_proportion

}

def calc_signals(input: SignalInput):
    return { signal_name: signal_func(input) for signal_name, signal_func in registered_signals.items() }
