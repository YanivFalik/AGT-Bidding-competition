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

dollar_utilization_threshold = 0.1
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

class PhaseSignal(Enum):
    EARLY_PHASE = auto()
    MID_PHASE = auto()
    LATE_PHASE = auto()

class BudgetRankSignal(Enum):
    RANK_1 = auto() # The player is rich
    RANK_2 = auto()
    RANK_3 = auto()
    RANK_4 = auto()
    RANK_5 = auto()

class SuccessRateSignal(Enum):
    LOW_SUCCESS_RATE = auto()
    MID_SUCCESS_RATE = auto()
    HIGH_SUCCESS_RATE = auto()

class OpponentModelingSignal(Enum):
    AGGRESSIVE = auto()
    TRUTHFUL = auto()
    CONSERVATIVE = auto()

class UtilityRankSignal(Enum):
    RANK_1 = auto()  # The player most utilized
    RANK_2 = auto()
    RANK_3 = auto()
    RANK_4 = auto()
    RANK_5 = auto()

class MaxItemGroupPosteriorSignal(Enum):
    LOW = auto()
    MIXED = auto()
    HIGH = auto()

@dataclass
class SignalInput:
    item_id: str
    our_team: str
    round_number: int
    our_budget: float
    competitor_budgets: Dict[str, float]
    competitor_items: Dict[str, list[str]]
    valuation_vector: Dict[str,float]
    posterior_vector: Dict[str,Belief]
    seen_items_and_prices: Dict[str,float]
    seen_items_ordered_by_round: list[str]
    current_utility: float

def get_closest_ith_order(belief: Belief, order: int):
    max_posterior = max(belief.p_low, belief.p_mixed, belief.p_high)
    closest_ith_order = high_order_statistics[order-1] if belief.p_high == max_posterior \
        else mixed_order_statistics[order-1] if belief.p_mixed == max_posterior \
        else low_order_statistics[order-1]
    return closest_ith_order

def get_expected_ith_order(belief: Belief, order: int):
    expected_ith_order = belief.p_low * low_order_statistics[order-1] \
                         + belief.p_mixed * mixed_order_statistics[order-1] \
                         + belief.p_high * high_order_statistics[order-1]
    return expected_ith_order

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
    expected_fourth_order = get_expected_ith_order(posteriors, 4)
    expected_utilization = (input.valuation_vector[input.item_id] - expected_fourth_order) / expected_fourth_order
    print(f"for item {input.item_id}, expected_fourth: {expected_fourth_order}, dollar utilization: {expected_utilization}")
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
        expected_fourth_order = get_expected_ith_order(belief, 4)
        return max(value - expected_fourth_order, 0.0)

    if input.round_number == 0:
        return ExpectedUtilityToRoundProportionSignal.CLOSE_TO_ONE

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


def signal_phase(input: SignalInput) -> PhaseSignal:
    """
    The function takes a signal input and check which phase we are in the game
    """
    if input.round_number < 6:
        return PhaseSignal.EARLY_PHASE
    elif input.round_number < 11:
        return PhaseSignal.MID_PHASE
    return PhaseSignal.LATE_PHASE

def signal_budget_rank(input: SignalInput) -> BudgetRankSignal:
    """
    The function takes a signal input and rate our budget base on other 4 players
    budget in the current round of the game
    """
    all_budget = list(input.competitor_budgets.values()) + [input.our_budget]
    sorted_budget = sorted(all_budget)
    our_rank_index = sorted_budget.index(input.our_budget)
    return list(BudgetRankSignal)[our_rank_index]

def signal_success_rate(input: SignalInput) -> SuccessRateSignal:
    """
    The function takes a signal input and look at all the seen item and our current
    utility and measure our success rate based on that : utility / our_valuation_for_seen_items
    """
    all_seen_items = input.seen_items_and_prices.keys() # list of all the items we have seen
    if not all_seen_items:
        return SuccessRateSignal.MID_SUCCESS_RATE

    total_value_seen = sum(input.valuation_vector[item_id] for item_id in all_seen_items)
    if total_value_seen == 0:
        return SuccessRateSignal.MID_SUCCESS_RATE

    utility_rate = input.current_utility / total_value_seen
    if utility_rate >= 0.25:
        return SuccessRateSignal.HIGH_SUCCESS_RATE
    elif utility_rate >= 0.10:
        return SuccessRateSignal.MID_SUCCESS_RATE
    return SuccessRateSignal.LOW_SUCCESS_RATE

opponent_modeling_look_behind = 2
opponent_modeling_threshold = 1
def signal_opponent_modeling(input: SignalInput) -> OpponentModelingSignal:
    if len(input.seen_items_and_prices) <= opponent_modeling_look_behind:
        last_items = input.seen_items_ordered_by_round[:]
    else:
        last_items = input.seen_items_ordered_by_round[-opponent_modeling_look_behind:]

    if len(last_items) == 0:
        return OpponentModelingSignal.TRUTHFUL

    diff_avg = sum([input.seen_items_and_prices[item_id] - get_closest_ith_order(input.posterior_vector[item_id], 4) \
                    for item_id in last_items]) / len(last_items)

    if -opponent_modeling_threshold <= diff_avg <= opponent_modeling_threshold:
        return OpponentModelingSignal.TRUTHFUL
    if diff_avg < -opponent_modeling_threshold:
        return OpponentModelingSignal.CONSERVATIVE
    return OpponentModelingSignal.AGGRESSIVE


def signal_utility_rank(input: SignalInput) -> UtilityRankSignal:
    competitor_utilities = {}
    for team, won_items in input.competitor_items.items():
        if team == input.our_team:
            continue
        competitor_utilities[team] = 0
        for won_item in won_items:
            utility = max(get_closest_ith_order(input.posterior_vector[won_item], 5) - input.seen_items_and_prices[won_item],0)
            competitor_utilities[team] += utility

    sorted_utilities = sorted(list(competitor_utilities.values()))
    our_utility = input.current_utility
    for i, utility in enumerate(sorted_utilities):
        if our_utility <= utility:
            if i == 0:
                return UtilityRankSignal.RANK_5
            elif i == 1:
                return UtilityRankSignal.RANK_4
            elif i == 2:
                return UtilityRankSignal.RANK_3
            elif i == 3:
                return UtilityRankSignal.RANK_2
            elif i == 4:
                return UtilityRankSignal.RANK_1
    return UtilityRankSignal.RANK_1

def signal_max_item_group_posterior(input: SignalInput) -> MaxItemGroupPosteriorSignal:
    posterior = input.posterior_vector[input.item_id]
    max_posterior = max(posterior.p_high, posterior.p_mixed, posterior.p_low)

    if posterior.p_mixed == max_posterior:
        return MaxItemGroupPosteriorSignal.MIXED
    if posterior.p_low == max_posterior:
        return MaxItemGroupPosteriorSignal.LOW
    if posterior.p_high == max_posterior:
        return MaxItemGroupPosteriorSignal.HIGH
    return MaxItemGroupPosteriorSignal.MIXED

registered_signals = {
    RelativeBudgetSignal.__name__ : signal_relative_budget,
    DollarUtilizationSignal.__name__ : signal_dollar_utilization,
    RemainingValueProportionSignal.__name__ : signal_remaining_value_proportion,
    ExpectedUtilityToRoundProportionSignal.__name__ : signal_expected_utility_to_round_proportion,
    PhaseSignal.__name__ : signal_phase,
    BudgetRankSignal.__name__ : signal_budget_rank,
    SuccessRateSignal.__name__ : signal_success_rate,
    OpponentModelingSignal.__name__ : signal_opponent_modeling,
    UtilityRankSignal.__name__ : signal_utility_rank,
    MaxItemGroupPosteriorSignal.__name__ : signal_max_item_group_posterior,
}

def calc_signals(input: SignalInput):
    signals = {}
    for signal_name, signal_func in registered_signals.items():
        print("Evaluating signal:", signal_name)
        signals[signal_name] = signal_func(input)
    return signals
