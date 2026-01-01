from dataclasses import dataclass
from enum import Enum
from typing import Dict

from src.signals import RelativeBudgetSignal, DollarUtilizationSignal


def calc_bid(item_value: float, signals: Dict[str, Enum]):
    print("Signals:\n"+str(signals))
    match signals[RelativeBudgetSignal.__name__]:

        case RelativeBudgetSignal.ABOVE_AVG:

            match signals[DollarUtilizationSignal.__name__]:

                case DollarUtilizationSignal.GEQ_THRESHOLD:

                    return 1.2 * item_value

                case DollarUtilizationSignal.LESS_THRESHOLD:

                    return 0.8 * item_value

        case RelativeBudgetSignal.AVG:

            match signals[DollarUtilizationSignal.__name__]:

                case DollarUtilizationSignal.GEQ_THRESHOLD:

                    return item_value

                case DollarUtilizationSignal.LESS_THRESHOLD:

                    return 0.6 * item_value


        case RelativeBudgetSignal.BELOW_AVG:

            match signals[DollarUtilizationSignal.__name__]:

                case DollarUtilizationSignal.GEQ_THRESHOLD:

                    return 0.8 * item_value

                case DollarUtilizationSignal.LESS_THRESHOLD:

                    return 0.2 * item_value
    return None