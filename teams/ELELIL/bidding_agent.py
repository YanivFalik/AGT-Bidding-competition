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
import os

from src.state_machine import calc_bid
from src.signals import calc_signals, SignalInput


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

from typing import Dict, List
import os 
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from item_beliefs import ItemBeliefs
from opponent_model import OpponentModeling

class BiddingAgent:

    
    def __init__(self, team_id: str, valuation_vector: Dict[str, float], 
                 budget: float, opponent_teams: List[str]):

        # Required attributes (DO NOT REMOVE)
        self.team_id = team_id
        self.valuation_vector = valuation_vector
        self.budget = budget
        self.initial_budget = budget
        self.opponent_teams = opponent_teams
        self.utility = 0
        self.items_won = []
        self.competitor_budgets = { opponent_team: self.budget for opponent_team in opponent_teams }
        self.seen_items_and_prices: Dict[str, float] = {}

        print("opponent teams:",opponent_teams)

        # Game state tracking
        self.rounds_completed = 0
        self.total_rounds = 15  # Always 15 rounds per game
        
        # ---------
        # TODO-----
        # ---------

        self.item_beliefs = ItemBeliefs(valuation_vector)
        self.opponent_model = OpponentModeling(opponent_teams)
        agent_logger.info(
            f"init: team={self.team_id}, budget={self.budget}, rounds_completed={self.rounds_completed}"
        )
        agent_logger.info(str(self.item_beliefs))

    
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


        self.item_beliefs.update_according_to_price(item_id, price_paid)
        agent_logger.info(
            f"Round {self.rounds_completed}, item {item_id}, winner {winning_team}, price_paid {price_paid}"
        )
        agent_logger.info(str(self.item_beliefs))

        self.opponent_model.update(winning_team, price_paid)
        
        return True
    
    def calc_shading(self, item_id: str) -> float: 
        return 1
    
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
        signal_dict = calc_signals(SignalInput(
            item_id = item_id,
            round_number = self.rounds_completed,
            our_budget = self.budget,
            competitor_budgets = self.competitor_budgets,
            valuation_vector =  self.valuation_vector,
            posterior_vector = self.item_beliefs.beliefs,
            seen_items_and_prices = self.seen_items_and_prices,
            current_utility = self.utility
        ))
        bid = calc_bid(self.valuation_vector[item_id], signal_dict)
        agent_logger.info(f"item: {item_id}, val: {self.valuation_vector[item_id]}, bid:, {bid}, signals: {signal_dict}")
        bid = max(0.0, min(bid, self.budget))
        
        return float(bid)
