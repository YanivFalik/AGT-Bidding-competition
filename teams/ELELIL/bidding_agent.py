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

from typing import Dict, List
import os 
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from item_beliefs import ItemBeliefs

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
        
        # Game state tracking
        self.rounds_completed = 0
        self.total_rounds = 15  # Always 15 rounds per game
        
        # ---------
        # TODO-----
        # ---------
        
        self.item_beliefs = ItemBeliefs(valuation_vector)
        print(self.item_beliefs)
    
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
        
        
        bid = my_valuation  
        bid = max(0.0, min(bid, self.budget))
        
        return float(bid)
    
