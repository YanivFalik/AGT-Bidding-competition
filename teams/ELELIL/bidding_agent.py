"""
AGT Competition - Student Agent Template
========================================

Team Name: ELELIL
Members: 
  - Alon Levy 313163958
  - Elizabeth Pinhasov 207501594
  - Yaniv Falik 314083551

Strategy Description:
At each round we calculate the probabilities for each item to be in the following groups: low, mixed or high, based on our valuation of the item.
After each round we update the probabilities for those groups based on the price paid for the last item sold.
Then we create a bidding function which tells us which strategy to use based on the probability, our valuation and the item group.
If we don’t think that we can win, we overbid so the winning team will pay more and spend more money.
If we think we can win we make a truthful bid - meaning we offer our true value for the item.

Key Features:
- Probability tracking using Bayes’ theorem
- Dynamic strategy according to expected fourth order statistic (i.e. expected second-highest bid)
- Guarding (overbidding) in cases of low valuations to make opponent pay more
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple

from teams.ELELIL.agent_logger import _make_agent_logger, beliefs_summary


@dataclass
class Belief:
    p_high: float
    p_mixed: float
    p_low: float

@dataclass
class SeenItemData:
    item_id: str
    winning_team: str
    price_paid: float
    round_seen: int
    potential_utility: float

agent_logger = _make_agent_logger()

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
    fourth_order = LOW_ORDER_STATISTICS[3] if group == GROUP_LOW \
        else (MIXED_ORDER_STATISTICS[3] if group == GROUP_MIXED else HIGH_ORDER_STATISTICS[3])
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

def linear_interpolation(start: float, end: float, t: float, t_max: float) -> float:
    alpha = (t_max - t) / t_max
    return max(start,min(start * alpha + end * (1 - alpha), end))

class BiddingAgent:

    def __init__(self, team_id: str, valuation_vector: Dict[str, float], 
                 budget: float, opponent_teams: List[str]):
        self.team_id = team_id
        self.valuation_vector = valuation_vector
        self.budget = budget
        self.init_budget = budget
        self.initial_budget = budget
        self.opponent_teams = opponent_teams
        self.items_won = []
        self.utility = 0

        # Game state tracking
        self.rounds_completed = 0
        self.total_rounds = 15  # Always 15 rounds per game

        self.seen_items: Dict[str, SeenItemData] = {}
        self.beliefs = get_posteriors_from_values(
                valuation_vector,
                Belief(TOTAL_HIGH / TOTAL_ITEMS, TOTAL_MIXED / TOTAL_ITEMS, TOTAL_LOW / TOTAL_ITEMS),
                [item_id for item_id, value in self.valuation_vector.items() if value >= VALUE_RANGE_HIGH[0]],
                [item_id for item_id, value in self.valuation_vector.items() if value <= VALUE_RANGE_LOW[1]]
        )
        agent_logger.info(
            f"init: team={self.team_id}, budget={self.budget}, rounds_completed={self.rounds_completed}"
        )
        agent_logger.info(beliefs_summary(self.valuation_vector, self.beliefs, self.seen_items, Belief(TOTAL_HIGH / TOTAL_ITEMS, TOTAL_MIXED / TOTAL_ITEMS, TOTAL_LOW / TOTAL_ITEMS)))

    
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

        # update history
        self.seen_items[item_id] = SeenItemData(
            item_id=item_id,
            winning_team=winning_team,
            price_paid=price_paid,
            round_seen=self.rounds_completed,
            potential_utility=max(0.0, self.valuation_vector[item_id] - price_paid)
        )

        # update beliefs of each value group
        self.beliefs, priors = get_updated_beliefs_according_to_price(item_id, price_paid, self.valuation_vector, self.beliefs, { item_id for item_id in self.seen_items })

        agent_logger.info(
            f"Round {self.rounds_completed}, item {item_id}, winner {winning_team}, price_paid {price_paid}"
        )
        agent_logger.info(beliefs_summary(self.valuation_vector, self.beliefs, self.seen_items, priors))
        
        return True
    
    def bidding_function(self, item_id: str) -> float:
        value = self.valuation_vector.get(item_id, 0)
        if value <= 0 or self.budget <= 0:
            return 0.0
        
        rounds_remaining = self.total_rounds - self.rounds_completed
        if rounds_remaining <= 0:
            return 0.0

        posteriors = self.beliefs[item_id]
        group, confidence = get_most_likely_group_and_confidence(posteriors)

        # Should we guard (overbid) or be truthful
        guarding = self.should_guard(value, group, confidence)
        if guarding:
            factor = self.calc_guard(value, group, confidence)
        else:
            factor = 1

        bid = value * factor

        # add shading for first rounds of game to not waste a lot of the budget at the beginning
        if not guarding and bid > HIGH_ORDER_STATISTICS[3]:
            round_shade = linear_interpolation(0.8, 1.0, self.rounds_completed, 8)
            bid *= round_shade

        agent_logger.info(f"item: {item_id}, val: {self.valuation_vector[item_id]}, guarding: {guarding}, factor: {factor} bid:, {bid}")

        return max(0.0, min(bid, self.budget))

    '''
    Guard = overbid in cases we don't expect to win in order to make the opponents pay more.
    Mostly, when we believe we are in the low part of an value group, we will bid close to the fourth order statistic (expected 2nd highest price)
    '''
    def calc_guard(self, value: float, item_group: str, confidence: float, margin: float = 0):
        if value < 5:
            return (LOW_ORDER_STATISTICS[3] - margin) / value

        if 10 < value:
            return (MIXED_ORDER_STATISTICS[3] - margin) / value

        # only raise to fourth order of high group if the item is from the high group with a certain confidence
        # to avoid winning the item and getting a negative utility
        if item_group == GROUP_HIGH and confidence >= 0.6:
            return (HIGH_ORDER_STATISTICS[3] - margin) / value

        return 1.4

    def should_guard(self, value, group: str, confidence: float) -> bool:
        if group == GROUP_MIXED and confidence >= 0.6:
            return value <= 13

        if confidence > 0.55:
            diff_from_fourth = get_diff_from_fourth_order(group, value)
            return diff_from_fourth < 0

        if 8 <= value < 10:
            return False

        if 10 < value <= MIXED_ORDER_STATISTICS[3]:
            return True

        if value >= 18:
            return False

        if 0 <= value < 5:
            return True

        return False

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
    beliefs = get_updated_posteriors_of_unseens(valuation_vector, beliefs, seen_items, priors, remainders, possible_highs, possible_lows)

    return beliefs, priors

def get_posteriors_from_values(items: Dict[str, float], priors: Belief, possible_highs: list[str], possible_lows: list[str]) -> Dict[str, Belief]:
    return {item_id: posterior_from_value(float(v), priors, possible_highs, possible_lows) for item_id, v in items.items()}

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

    # edge case. happened once.
    if w_sum == 0:
        return priors

    # create new belief with ( P(T=high|v), P(T=mixed|v), P(T=low|v) )
    return Belief(w_high / w_sum, w_mixed / w_sum, w_low / w_sum)


MIXED_ENSURANCE_THRESHOLD = 0.1
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
def get_posterior_with_price(item_id: str, value: float, price_paid: float, posterior_without_price: Belief) -> Belief:
    """
    For n i.i.d. samples from Uniform[0,1], the density of the k-th order statistic (k-th item in increasing order) is:
    f_k(y) = n!/((k-1)!(n-k)!)y^(k-1)(1-y)^(n-k)
    Where:
     - y = the value of the density function (which is also the probability to be lower than y because the samples are from Uniform[0,1]).
     - k-1 = items below y
     - n-k = items above y
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

    expected_high_remainder = max(0.0, TOTAL_HIGH - expected_used_high)
    expected_mixed_remainder = max(0.0, TOTAL_MIXED - expected_used_mixed)
    expected_low_remainder = max(0.0, TOTAL_LOW - expected_used_low)
    return Belief(expected_high_remainder, expected_mixed_remainder, expected_low_remainder)


"""
In order to guess the remaining number of items in a value group t we can calculate the expected remainder as follows:
    E[remaining items in t] = E[expected items remaining] - E[removed items in t]
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

def get_updated_posteriors_of_unseens(
        valuation_vector: Dict[str, float],
        beliefs: Dict[str, Belief],
        seen_items: set[str],
        priors: Belief,
        expected_remainders: Belief,
        possible_highs: list[str],
        possible_lows: list[str]) -> Dict[str, Belief]:
    unseen = {k:v for k,v in valuation_vector.items() if k not in seen_items}
    posteriors = get_posteriors_from_values(unseen, priors, possible_highs, possible_lows)

    # normalize posteriors such that sum_i(P(T_i=t)) = E[remaining items in t]
    normalized_posteriors = get_normalized_posteriors(posteriors, expected_remainders)
    return { item_id : normalized_posteriors[item_id] if item_id in normalized_posteriors else beliefs[item_id] \
             for item_id in valuation_vector }

""" ============================================================================================================
=============================== END BELIEF CALCULATIONS ========================================================
============================================================================================================ """

