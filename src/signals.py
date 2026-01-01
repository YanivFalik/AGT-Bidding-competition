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
from typing import Dict

from teams.ELELIL.item_beliefs import Belief

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

@dataclass
class SignalInput:
    item_id: str
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

registered_signals = {
    RelativeBudgetSignal.__name__ : signal_relative_budget,
    DollarUtilizationSignal.__name__ : signal_dollar_utilization,
    RemainingValueProportionSignal.__name__ : signal_remaining_value_proportion,
}

def calc_signals(input: SignalInput):
    return { signal_name: signal_func(input) for signal_name, signal_func in registered_signals.items() }
