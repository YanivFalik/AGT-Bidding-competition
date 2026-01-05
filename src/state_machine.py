from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from unittest import case
from src.signals import *

def win_or_guard(item_value):
    # if we are between [8,10], overbid.
    # If the item is mixed, there is probably a higher bidder and we pushed him to pay more
    # If the item is low, we are probably the highest bidder and will win the item for a lower price
    if low_order_statistics[3] + 1 <= item_value <= 10:
        return mixed_order_statistics[3] - 0.5

    if mixed_order_statistics[3] <= item_value <= high_order_statistics[3]:
        return (high_order_statistics[3] - 0.5) * 0.9

    return item_value * 0.8



def state_machine(item_value: float, posterior: Belief, signals: Dict[str, Enum]):
    """
    Signal-driven bid multiplier policy (only state machine changed).
    Returns a bid in [0, item_value].
    """

    # --- base multiplier starts neutral ---
    m = 1.0

    # 1) Item-level desirability: DollarUtilization
    match signals[DollarUtilizationSignal.__name__]:
        case DollarUtilizationSignal.GEQ_THRESHOLD:
            m *= 1.40
        case DollarUtilizationSignal.LESS_THRESHOLD:
            m *= 0.60

    # 2) Are we behind/at/ahead of schedule? (utility vs time)
    match signals[ExpectedUtilityToRoundProportionSignal.__name__]:
        case ExpectedUtilityToRoundProportionSignal.BELOW_ONE:
            m *= 1.3
        case ExpectedUtilityToRoundProportionSignal.CLOSE_TO_ONE:
            m *= 1.00
        case ExpectedUtilityToRoundProportionSignal.ABOVE_ONE:
            m *= 1.00

    # 3) Remaining value pressure
    match signals[RemainingValueProportionSignal.__name__]:
        case RemainingValueProportionSignal.LOW:
            m *= 1.15
        case RemainingValueProportionSignal.MID:
            m *= 1.00
        case RemainingValueProportionSignal.HIGH:
            m *= 0.95

    # 4) Opponent market behavior
    match signals[OpponentModelingSignal.__name__]:
        case OpponentModelingSignal.AGGRESSIVE:
            m *= 1.10
        case OpponentModelingSignal.TRUTHFUL:
            m *= 1.00
        case OpponentModelingSignal.CONSERVATIVE:
            m *= 0.92

    # 5) Budget position (use both rank + relative budget, gently)
    match signals[BudgetRankSignal.__name__]:
        case BudgetRankSignal.RANK_1:
            m *= 1.2
        case BudgetRankSignal.RANK_2:
            m *= 1.1
        case BudgetRankSignal.RANK_3:
            m *= 1.00
        case BudgetRankSignal.RANK_4:
            m *= 1.0
        case BudgetRankSignal.RANK_5:
            m *= 1.00

    match signals[RelativeBudgetSignal.__name__]:
        case RelativeBudgetSignal.ABOVE_AVG:
            m *= 1.03
        case RelativeBudgetSignal.AVG:
            m *= 1.00
        case RelativeBudgetSignal.BELOW_AVG:
            m *= 0.97

    # 6) How well are we doing so far?
    match signals[SuccessRateSignal.__name__]:
        case SuccessRateSignal.HIGH_SUCCESS_RATE:
            m *= 0.96
        case SuccessRateSignal.MID_SUCCESS_RATE:
            m *= 1.00
        case SuccessRateSignal.LOW_SUCCESS_RATE:
            m *= 1.2

    # 7) Utility rank: if we’re behind others, push; if leading, shade
    match signals[UtilityRankSignal.__name__]:
        case UtilityRankSignal.RANK_5 | UtilityRankSignal.RANK_4:
            m *= 1.2
        case UtilityRankSignal.RANK_3:
            m *= 1.00
        case UtilityRankSignal.RANK_2 | UtilityRankSignal.RANK_1:
            m *= 0.94

    # 8) Phase-specific shaping (endgame: be more decisive)
    match signals[PhaseSignal.__name__]:
        case PhaseSignal.EARLY_PHASE:
            m *= 0.95
        case PhaseSignal.MID_PHASE:
            m *= 1.00
        case PhaseSignal.LATE_PHASE:
            m *= 1.12

    # 9) If item is "LOW" group and we're late + not desperate, shade (avoid overpaying junk late)
    if signals[MaxItemGroupPosteriorSignal.__name__] == MaxItemGroupPosteriorSignal.LOW:
        if signals[PhaseSignal.__name__] == PhaseSignal.LATE_PHASE and \
           signals[ExpectedUtilityToRoundProportionSignal.__name__] != ExpectedUtilityToRoundProportionSignal.BELOW_ONE:
            m *= 0.85

    m = min(max(m, 1), 1.2)
    # final bid (never exceed value)
    bid = m * item_value
    return min(item_value, max(0.0, bid))


def calc_bid(item_value: float, posterior: Belief, signals: Dict[str, Enum]):
    print("Signals:\n"+str(signals))
    sorted_posteriors = sorted([posterior.p_high, posterior.p_mixed, posterior.p_low])
    highest_post = sorted_posteriors[-1]
    second_highest_post = sorted_posteriors[-2]

    if highest_post - second_highest_post <= 0.25:
        return win_or_guard(item_value)

    return state_machine(item_value, posterior, signals)



