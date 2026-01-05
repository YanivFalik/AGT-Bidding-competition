"""
AGT Competition - Student Agent Template
========================================

Team Name: [ELELIL]
Members: 
  - [Student 1 Name and ID]
  - [Student 2 Name and ID]
  - [Student 3 Name and ID]

Strategy Description:
[Brief description of your bidding strategy]

Key Features:
- [Feature 1]
- [Feature 2]
- [Feature 3]
"""

import logging
import math
import os

from src.config import LOW_VALUE_ITEMS
# from src.state_machine import calc_bid
from src.signals import calc_signals, SignalInput, low_order_statistics
from dataclasses import dataclass

from teams.ELELIL.belief_functional import Belief


@dataclass
class SeenItemData:
    item_id: str
    winning_team: str
    price_paid: float
    round_seen: int
    potential_utility: float

def _make_agent_logger(name: str = "agent_tmp_logger") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # IMPORTANT: don't pass messages to root/main handlers

    # Add handler only once (avoid duplicate lines if module is reloaded)
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        log_path = os.path.join(os.path.dirname(__file__), "tmp.log")
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(fh)

    return logger

agent_logger = _make_agent_logger()

from typing import Dict, List, Tuple
import os 
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from item_beliefs import ItemBeliefs
from opponent_model import OpponentModeling

TOTAL_ROUNDS = 15

HIGH_ORDER_STATISTICS = [11 + 2 / 3, 13 + 1 / 3, 15, 16 + 2 / 3, 18 + 1 / 3]
MIXED_ORDER_STATISTICS = [4 + 1 / 6, 7 + 1 / 3, 10.5, 13 + 2 / 3, 16 + 5 / 6]
LOW_ORDER_STATISTICS = [2.5, 4, 5.5, 7, 8.5]

GROUP_HIGH = 'high'
GROUP_MIXED = 'mixed'
GROUP_LOW = 'low'

STRATEGY_WIN = 'win'
STRATEGY_GUARD = 'guard'
STRATEGY_TRUTHFUL = 'truthful'

def get_diff_from_fourth_order(group: str, item_value: float):
    fourth_order = LOW_ORDER_STATISTICS[3] if group == 'low' \
        else (MIXED_ORDER_STATISTICS[3] if group == 'mixed' else HIGH_ORDER_STATISTICS[3])
    return item_value - fourth_order

def get_most_likely_group_and_confidence(posteriors: Belief):
    max_posterior = max(posteriors.p_high, posteriors.p_low, posteriors.p_mixed)
    group = ''
    if posteriors.p_mixed == max_posterior:
        group = GROUP_MIXED
    elif posteriors.p_high == max_posterior:
        group = GROUP_HIGH
    else:
        group = GROUP_LOW
    return group, max_posterior

class BiddingAgent:

    
    def __init__(self, team_id: str, valuation_vector: Dict[str, float], 
                 budget: float, opponent_teams: List[str]):

        # Required attributes (DO NOT REMOVE)
        self.team_id = team_id
        self.valuation_vector = valuation_vector
        self.budget = budget
        self.init_budget = budget
        self.initial_budget = budget
        self.opponent_teams = opponent_teams
        self.utility = 0
        self.items_won = []
        self.lost_seen_items_by_round = []
        self.competitor_budgets = { opponent_team: self.budget for opponent_team in opponent_teams }
        self.competitor_items = { opponent_team: [] for opponent_team in opponent_teams }
        self.seen_items: Dict[str, SeenItemData] = {}
        self.seen_items_and_prices: Dict[str, float] = {}
        self.bid = 0


        # Game state tracking
        self.rounds_completed = 0
        self.total_rounds = 15  # Always 15 rounds per game
        
        # ---------
        # TODO-----
        # ---------
        self.beliefs = get_posteriors_from_values(valuation_vector, Belief(TOTAL_HIGH / TOTAL_ITEMS, TOTAL_MIXED / TOTAL_ITEMS, TOTAL_LOW / TOTAL_ITEMS))
        self.opponent_model = OpponentModeling(opponent_teams)
        agent_logger.info(
            f"init: team={self.team_id}, budget={self.budget}, rounds_completed={self.rounds_completed}"
        )
        agent_logger.info(beliefs_summary(self.valuation_vector, self.beliefs, self.seen_items_and_prices, Belief(TOTAL_HIGH / TOTAL_ITEMS, TOTAL_MIXED / TOTAL_ITEMS, TOTAL_LOW / TOTAL_ITEMS)))

    
    def _update_available_budget(self, item_id: str, winning_team: str, 
                                 price_paid: float):
        """
        DO NOT MODIFY 
        """
        if winning_team == self.team_id:
            self.budget -= price_paid
            self.items_won.append(item_id)
    
    def update_after_each_round(self, item_id: str, winning_team: str, 
                                price_paid: float):
        # System updates (DO NOT REMOVE)
        self._update_available_budget(item_id, winning_team, price_paid)
        
        if winning_team == self.team_id:
            self.utility += (self.valuation_vector[item_id] - price_paid)
        
        self.rounds_completed += 1
        
        # ============================================================
        # TODO: implement 
        # ============================================================
        self.seen_items_and_prices[item_id] = price_paid
        if winning_team != self.team_id:
            self.competitor_budgets[winning_team] -= price_paid
            self.competitor_items[winning_team].append(item_id)
            self.lost_seen_items_by_round.append(item_id)
            self.seen_items[item_id] = SeenItemData(
                item_id=item_id,
                winning_team=winning_team,
                price_paid=price_paid,
                round_seen=self.rounds_completed,
                potential_utility=max(0.0, self.valuation_vector[item_id] - price_paid)
            )


        self.beliefs, priors = get_updated_beliefs_according_to_price(item_id, price_paid, self.valuation_vector, self.beliefs, { item_id for item_id in self.seen_items_and_prices })
        agent_logger.info(
            f"Round {self.rounds_completed}, item {item_id}, winner {winning_team}, price_paid {price_paid}, {('GUARDED' if math.fabs(self.bid-price_paid) < 0.01 else '')}"
        )
        agent_logger.info(beliefs_summary(self.valuation_vector, self.beliefs, self.seen_items_and_prices, priors))

        self.opponent_model.update(winning_team, price_paid)
        
        return True
    
    def bidding_function(self, item_id: str) -> float:
        my_valuation = self.valuation_vector.get(item_id, 0)
        
        if my_valuation <= 0 or self.budget <= 0:
            return 0.0
        
        rounds_remaining = self.total_rounds - self.rounds_completed
        if rounds_remaining <= 0:
            return 0.0
        
        # ============================================================
        # TODO: IMPLEMENT YOUR BIDDING STRATEGY HERE
        # ============================================================
        # signal_dict = calc_signals(SignalInput(
        #     item_id = item_id,
        #     our_team=self.team_id,
        #     round_number = self.rounds_completed,
        #     our_budget = self.budget,
        #     competitor_budgets = self.competitor_budgets,
        #     competitor_items=self.competitor_items,
        #     valuation_vector =  self.valuation_vector,
        #     posterior_vector = self.beliefs,
        #     seen_items_and_prices = self.seen_items_and_prices,
        #     current_utility = self.utility,
        #     seen_items_ordered_by_round=self.lost_seen_items_by_round
        # ))
        #
        # bid = calc_bid(self.valuation_vector[item_id], self.beliefs[item_id], self.rounds_completed, signal_dict)

        value = self.valuation_vector[item_id]
        posteriors = self.beliefs[item_id]

        group, confidence = get_most_likely_group_and_confidence(posteriors)
        strategy = self.calc_strategy(value, group, confidence)
        if strategy == STRATEGY_GUARD:
            factor = self.calc_guard(value, posteriors, group, confidence)
        elif strategy == STRATEGY_WIN:
            factor = self.calc_win(value, posteriors, group)
        else:
            factor = 1

        bid = value * factor
        self.bid = bid

        if strategy != STRATEGY_GUARD and bid > 16:
            alpha = (8 - self.rounds_completed) / 8
            round_shade = min(0.8 * alpha + 1 * (1-alpha), 1.0)
            bid *= round_shade

        agent_logger.info(f"item: {item_id}, val: {self.valuation_vector[item_id]}, strategy: {strategy}, factor: {factor} bid:, {bid}")

        return max(0.0, min(bid, self.budget))

    def calc_guard(self, value: float, posteriors: Belief, item_group: str, confidence: float, margin: float = 0):
        if value < 5:
            return (LOW_ORDER_STATISTICS[3] - margin) / value

        if 10 < value:
            return (MIXED_ORDER_STATISTICS[3] - margin) / value

        if item_group == GROUP_HIGH and confidence >= 0.6:
            return (HIGH_ORDER_STATISTICS[3] - margin) / value

        return 1.4

    def calc_win(self, value: float, posteriors: Belief, item_group: str):
        def get_target_spending(round_proportion: float, epsilon: float = 0.5):
            aggressiveness_by_round = math.pow(round_proportion, epsilon)
            spend_budget_fraction = (self.init_budget - self.budget) / self.init_budget
            return max(spend_budget_fraction / aggressiveness_by_round, 0.1)

        def get_average_lost_utility():
            lost_utilities = [item.potential_utility for item in self.seen_items.values() if item.potential_utility > 0]
            if len(lost_utilities) == 0:
                return 0
            return sum(lost_utilities) / len(lost_utilities)

        proportional_round = (self.rounds_completed+1) / TOTAL_ROUNDS

        factor = 1.0
        if get_target_spending(proportional_round, 0.8) < 1:
            factor *= 1.2
        else:
            factor *= 1

        if get_average_lost_utility() > 1:
            factor *= 1.3
        elif get_average_lost_utility() > 0:
            factor *= 1.2
        else:
            factor *= 1

        agent_logger.info(f"item: {item_group}, factor: {factor}, target spending: {get_target_spending(proportional_round, 0.8)}, 'average loss: {get_average_lost_utility()}")
        factor = min(max(factor, 0.8), 1.4)
        if item_group in [GROUP_HIGH]:
            factor = 1 # if high, be truthful

        return factor

    # 'win', 'guard', 'truthful'
    def calc_strategy(self, value, group: str, confidence: float):
        if group == GROUP_MIXED and confidence >= 0.6:
            if value > 13:
                return 'win'
            else:
                return 'guard'

        if confidence > 0.55:
            diff_from_fourth = get_diff_from_fourth_order(group, value)
            return 'guard' if diff_from_fourth < 0 else 'win'

        if 8 <= value < 10:
            return 'win'

        if 10 < value <= MIXED_ORDER_STATISTICS[3]:
            return 'guard'

        if value >= 18:
            return 'win'

        if 0 <= value < 5:
            return 'guard'

        return 'truthful'









""" ============================================================================================================
================================= BELIEF CALCULATIONS ==========================================================
============================================================================================================ """

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

def get_updated_beliefs_according_to_price(
        item_id: str,
        price_paid: float,
        valuation_vector: Dict[str, float],
        beliefs: Dict[str, Belief],
        seen_items: set[str]
    ):
    # For price p of item i with value v:
    # calculate P(T_i=t | v,p) for every t in [LOW,MIXED,HIGH]
    beliefs[item_id] = get_posterior_with_price(item_id, valuation_vector[item_id], price_paid, beliefs[item_id])

    # SPECIAL CASE HANDLING: in case of P(T_i=MIXED)=1, remove i from the list of possible LOW/HIGH items
    possible_highs, possible_lows = get_group_possible_candidates(valuation_vector, {})

    # Calculate E[items remaining in group t] for every t in [LOW,MIXED,HIGH]
    remainders = get_expected_remainders_from_seen(beliefs, seen_items)

    # Update P(T_i=t) to be E[items remaining in t]/E[items remaining] for every t in [LOW,MIXED,HIGH]
    priors = get_global_priors(remainders)

    # Update P(T_j | v) for all remaining items
    beliefs = get_updated_posteriors_of_unseens(valuation_vector, beliefs, seen_items, priors, remainders)

    return beliefs, priors

def get_posteriors_from_values(items: Dict[str, float], priors: Belief) -> Dict[str, Belief]:
    return {item_id: posterior_from_value(float(v), priors) for item_id, v in items.items()}

"""
Bayes’ rule for continuous observations:
Given PDF f, item i with value v, value group T (high, mixed, low) we get:

P(T_i=t|v) = P(T_i=t)f(v|T_i=t) / sum_t'(P(T_i=t')f(v|T_i=t'))

Where P(T_i=t) is the global prior updated each round
"""
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


"""
In order to guess the remaining number of items in a value group t we can calculate the expected remainder as follows:
    E[remaining items in t] = TOTAL_T - E[removed items in t]
Where:
    E[removed items in t] = sum_{i : removed items}P(T_i = t)

Now we can estimate the probability of a random item i being in value group t as:
    P(T_i=t) = E[remaining items in t]/total remaining items
"""
def get_global_priors(expected_remainders: Belief) -> Belief:
    expected_items_left = expected_remainders.p_high + expected_remainders.p_mixed + expected_remainders.p_low
    if expected_items_left <= 0:
        return Belief(0,0,0)
    return Belief(
        expected_remainders.p_high / expected_items_left,
        expected_remainders.p_mixed / expected_items_left,
        expected_remainders.p_low / expected_items_left
    )

"""
Update all posteriors such that sum_i(P(T_i=t)) = E[items remaining it t]
This is done by calculating factors 
"""
def get_normalized_posteriors(posteriors: Dict[str, Belief], expected_remainders: Belief) -> Dict[str, Belief]:
    cumulative_high_prob = sum(belief.p_high for belief in posteriors.values())
    cumulative_mixed_prob = sum(belief.p_mixed for belief in posteriors.values())
    cumulative_low_prob = sum(belief.p_low for belief in posteriors.values())

    high_factor = expected_remainders.p_high / cumulative_high_prob if cumulative_high_prob > 0 else 1.0
    mixed_factor = expected_remainders.p_mixed / cumulative_mixed_prob if cumulative_mixed_prob > 0 else 1.0
    low_factor = expected_remainders.p_low / cumulative_low_prob if cumulative_low_prob > 0 else 1.0

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

def get_updated_posteriors_of_unseens(valuation_vector: Dict[str, float], beliefs: Dict[str, Belief], seen_items: set[str], priors: Belief, expected_remainders: Belief) -> Dict[str, Belief]:
    unseen = {k:v for k,v in valuation_vector.items() if k not in seen_items}
    posteriors = get_posteriors_from_values(unseen, priors)

    # normalize posteriors such that sum_i(P(T_i=t)) = E[remaining items in t]
    normalized_posteriors = get_normalized_posteriors(posteriors, expected_remainders)
    return { item_id : normalized_posteriors[item_id] if item_id in normalized_posteriors else beliefs[item_id] \
             for item_id in valuation_vector }

def beliefs_summary(valuation_vector: Dict[str, float], beliefs: Dict[str, Belief], seen_items, priors: Belief, digits: int = 3) -> str:
    fmt = f"{{:.{digits}f}}"
    items = sorted(valuation_vector.items(), key=lambda x: x[1], reverse=True)

    lines = []
    lines.append(
        "Item Beliefs (initial or updated)\n"
        "--------------------------------\n"
        "item_id   value   P(High)   P(Mixed)    P(Low)"
    )
    for item_id, v in items:
        b = beliefs[item_id]
        lines.append(
            f"{item_id:<9} {v:>5.1f}   "
            f"{fmt.format(b.p_high):>7}   {fmt.format(b.p_mixed):>7}   {fmt.format(b.p_low):>7}"
            + ("   [SEEN]" if item_id in seen_items else "")
        )
    lines.append(
        f"\nGlobal priors for unseen items: "
        f"High={priors.p_high:.3f}, MIXED={priors.p_mixed:.3f}, Low={priors.p_low:.3f}"
    )
    return "\n".join(lines)

""" ============================================================================================================
=============================== END BELIEF CALCULATIONS ========================================================
============================================================================================================ """

