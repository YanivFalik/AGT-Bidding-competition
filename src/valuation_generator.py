"""
Valuation Generator for AGT Competition
Generates item valuations according to competition rules
"""

import numpy as np
from typing import Dict, List, Tuple
from src.config import (
    K_TOTAL_ITEMS, HIGH_VALUE_ITEMS, LOW_VALUE_ITEMS, MIXED_VALUE_ITEMS,
    HIGH_VALUE_RANGE, LOW_VALUE_RANGE, MIXED_VALUE_RANGE, ITEM_ID_FORMAT, RANDOM_SEED
)


class ValuationGenerator:
    """
    Generates valuation vectors for teams according to competition specifications.
    
    Distribution:
    - 6 items: High-value for all teams (U[10,20])
    - 4 items: Low-value for all teams (U[1,10])
    - 10 items: Mixed values (U[1,20])
    """
    
    def __init__(self, random_seed: int = None):
        """
        Initialize valuation generator.
        
        Args:
            random_seed: Optional seed for reproducibility
        """
        self.random_seed = random_seed if random_seed is not None else RANDOM_SEED
        if self.random_seed is not None:
            np.random.seed(self.random_seed)
        
        # Verify configuration
        assert HIGH_VALUE_ITEMS + LOW_VALUE_ITEMS + MIXED_VALUE_ITEMS == K_TOTAL_ITEMS, \
            "Item categories must sum to K_TOTAL_ITEMS"
    
    def _generate_item_categories(self) -> Tuple[List[str], List[str], List[str]]:
        """
        Randomly assign items to categories (high, low, mixed).
        This is done once per game so all teams have consistent categorization.
        
        Returns:
            Tuple of (high_value_items, low_value_items, mixed_value_items)
        """
        all_items = [ITEM_ID_FORMAT.format(i) for i in range(K_TOTAL_ITEMS)]
        np.random.shuffle(all_items)
        
        high_value_items = all_items[:HIGH_VALUE_ITEMS]
        low_value_items = all_items[HIGH_VALUE_ITEMS:HIGH_VALUE_ITEMS + LOW_VALUE_ITEMS]
        mixed_value_items = all_items[HIGH_VALUE_ITEMS + LOW_VALUE_ITEMS:]
        
        return high_value_items, low_value_items, mixed_value_items
    
    def generate_valuation_vector(self, team_id: str, 
                                  high_items: List[str],
                                  low_items: List[str],
                                  mixed_items: List[str]) -> Dict[str, float]:
        """
        Generate a valuation vector for a single team.
        
        Args:
            team_id: Unique team identifier
            high_items: List of item IDs that are high-value for all teams
            low_items: List of item IDs that are low-value for all teams
            mixed_items: List of item IDs with mixed values
        
        Returns:
            Dictionary mapping item_id to valuation
        """
        valuation_vector = {}
        
        # High-value items (same items for all teams, but different values)
        for item_id in high_items:
            valuation_vector[item_id] = np.random.uniform(*HIGH_VALUE_RANGE)
        
        # Low-value items (same items for all teams, but different values)
        for item_id in low_items:
            valuation_vector[item_id] = np.random.uniform(*LOW_VALUE_RANGE)
        
        # Mixed-value items (can be high or low for different teams)
        for item_id in mixed_items:
            valuation_vector[item_id] = np.random.uniform(*MIXED_VALUE_RANGE)
        
        return valuation_vector
    
    def generate_arena_valuations(self, team_ids: List[str]) -> Tuple[Dict[str, Dict[str, float]], 
                                                                       Tuple[List[str], List[str], List[str]]]:
        """
        Generate valuations for all teams in an arena.
        All teams get the same item categorization but different values.
        
        Args:
            team_ids: List of team IDs in the arena
        
        Returns:
            Tuple of (valuations_dict, item_categories)
            - valuations_dict: {team_id: {item_id: valuation}}
            - item_categories: (high_items, low_items, mixed_items)
        """
        # Determine item categories (consistent for all teams)
        high_items, low_items, mixed_items = self._generate_item_categories()
        
        # Generate valuations for each team
        valuations = {}
        for team_id in team_ids:
            valuations[team_id] = self.generate_valuation_vector(
                team_id, high_items, low_items, mixed_items
            )
            print('valuation of', team_id, ':', valuations[team_id])
        
        return valuations, (high_items, low_items, mixed_items)
    
    def get_random_auction_sequence(self, num_items: int = None) -> List[str]:
        """
        Select and shuffle random items for auction sequence.
        
        Args:
            num_items: Number of items to auction (default from config)
        
        Returns:
            List of item IDs in random order
        """
        from src.config import T_AUCTION_ROUNDS
        if num_items is None:
            num_items = T_AUCTION_ROUNDS
        
        all_items = [ITEM_ID_FORMAT.format(i) for i in range(K_TOTAL_ITEMS)]
        selected_items = np.random.choice(all_items, size=num_items, replace=False).tolist()
        np.random.shuffle(selected_items)
        
        return selected_items
