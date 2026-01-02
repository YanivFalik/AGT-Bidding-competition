from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional
from unittest import case
from src.signals import *

def win_or_guard(item_value):
    if low_order_statistics[3] <= item_value <= 10:
        return mixed_order_statistics[3] - 0.5

    if mixed_order_statistics[3] <= item_value <= high_order_statistics[3]:
        return (high_order_statistics[3] - 0.5) * 0.9

    return item_value



def state_machine(item_value: float, posterior: Belief, signals: Dict[str, Enum]):
    match signals[PhaseSignal.__name__]:

        case PhaseSignal.EARLY_PHASE:

            # match signals[DollarUtilizationSignal.__name__]:
            #     case DollarUtilizationSignal.LESS_THRESHOLD:
            #         return 0.8 * item_value
            #
            #     case DollarUtilizationSignal.GEQ_THRESHOLD:
            #
            #         return 1.2 * item_value

        case PhaseSignal.MID_PHASE:

            match signals[UtilityRankSignal.__name__]:
                case UtilityRankSignal.RANK_5 | UtilityRankSignal.RANK_4 | UtilityRankSignal.RANK_3:

                    # match signals[DollarUtilizationSignal.__name__]:
                    #     case DollarUtilizationSignal.LESS_THRESHOLD:
                    #
                    #         return 1 * item_value
                    #
                    #     case DollarUtilizationSignal.GEQ_THRESHOLD:
                    #
                    #         return 1.4 * item_value

                case UtilityRankSignal.RANK_2 | UtilityRankSignal.RANK_1:

                    # match signals[DollarUtilizationSignal.__name__]:
                    #     case DollarUtilizationSignal.LESS_THRESHOLD:
                    #
                    #         return 0.8 * item_value
                    #
                    #     case DollarUtilizationSignal.GEQ_THRESHOLD:
                    #
                    #         return 1.2 * item_value

        case PhaseSignal.LATE_PHASE:

            match signals[RelativeBudgetSignal.__name__]:
                case RelativeBudgetSignal.BELOW_AVG:

                    return 1.1 * item_value

                case RelativeBudgetSignal.AVG:

                    return 1 * item_value

                case RelativeBudgetSignal.ABOVE_AVG:

                    # match signals[DollarUtilizationSignal.__name__]:
                    #     case DollarUtilizationSignal.LESS_THRESHOLD:
                    #
                    #         return 1 * item_value
                    #
                    #     case DollarUtilizationSignal.GEQ_THRESHOLD:
                    #
                    #         return 1.2 * item_value

    match signals[RelativeBudgetSignal.__name__]:

        case RelativeBudgetSignal.ABOVE_AVG:

            # match signals[DollarUtilizationSignal.__name__]:
            #
            #     case DollarUtilizationSignal.GEQ_THRESHOLD:
            #
            #         return 1.2 * item_value
            #
            #     case DollarUtilizationSignal.LESS_THRESHOLD:
            #
            #         return 0.8 * item_value

        case RelativeBudgetSignal.AVG:

            # match signals[DollarUtilizationSignal.__name__]:
            #
            #     case DollarUtilizationSignal.GEQ_THRESHOLD:
            #
            #         return item_value
            #
            #     case DollarUtilizationSignal.LESS_THRESHOLD:
            #
            #         return 0.6 * item_value

        case RelativeBudgetSignal.BELOW_AVG:

            # match signals[DollarUtilizationSignal.__name__]:
            #
            #     case DollarUtilizationSignal.GEQ_THRESHOLD:
            #
            #         return 0.8 * item_value
            #
            #     case DollarUtilizationSignal.LESS_THRESHOLD:
            #
            #         return 0.2 * item_value
    return None


def calc_bid(item_value: float, posterior: Belief, signals: Dict[str, Enum]):
    print("Signals:\n"+str(signals))

    sorted_posteriors = sorted([posterior.p_high, posterior.p_mixed, posterior.p_low])
    highest_post = sorted_posteriors[-1]
    second_highest_post = sorted_posteriors[-2]

    if highest_post - second_highest_post <= 0.2:
        return win_or_guard(item_value)

    return state_machine(item_value, posterior, signals)



