from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class Belief:
    """Posterior over item type."""
    p_high: float
    p_all: float
    p_low: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.p_high, self.p_all, self.p_low)


class ItemBeliefs:
    """
    Maintains per-item beliefs about the hidden item type T ∈ {High, All, Low}.

    Initialization uses only your valuation v_i and the known priors + uniform densities.
    Later, you can update beliefs for an auctioned item using the observed price (placeholder).
    """

    # Fixed priors from the competition description
    PRIOR_HIGH = 6 / 20
    PRIOR_LOW = 4 / 20
    PRIOR_ALL = 10 / 20

    def __init__(self, valuation_vector: Dict[str, float]):
        self.valuation_vector = dict(valuation_vector)

        # beliefs[item_id] = Belief(p_high, p_all, p_low)
        self.beliefs: Dict[str, Belief] = {}

        # Precompute initial beliefs from v_i alone
        for item_id, v in self.valuation_vector.items():
            self.beliefs[item_id] = self._initial_posterior_from_value(float(v))

    def get(self, item_id: str) -> Belief:
        """Return current belief for item_id."""
        return self.beliefs[item_id]

    def _initial_posterior_from_value(self, v: float) -> Belief:
        """
        Compute P(T | v) using Bayes with uniform densities.
        NOTE: Because distributions are uniform, posterior is piecewise-constant:
          - v in [1,10): only Low and All possible
          - v in (10,20]: only High and All possible
          - v == 10: all three possible (boundary case)
        """

        # Densities (uniform) on supports:
        # High: U[10,20] => 1/10 on [10,20]
        # Low:  U[1,10]  => 1/9  on [1,10]
        # All:  U[1,20]  => 1/19 on [1,20]
        f_high = (1.0 / 10.0) if (10.0 <= v <= 20.0) else 0.0
        f_low = (1.0 / 9.0) if (1.0 <= v <= 10.0) else 0.0
        f_all = (1.0 / 19.0) if (1.0 <= v <= 20.0) else 0.0

        w_high = self.PRIOR_HIGH * f_high
        w_low = self.PRIOR_LOW * f_low
        w_all = self.PRIOR_ALL * f_all

        Z = w_high + w_low + w_all
        if Z <= 0:
            # Shouldn't happen if v is in [1,20], but keep safe.
            return Belief(0.0, 1.0, 0.0)

        return Belief(
            p_high=w_high / Z,
            p_all=w_all / Z,
            p_low=w_low / Z,
        )

    def update_with_price(
        self,
        item_id: str,
        price_paid: float,
        winning_team: Optional[str] = None,
        my_bid: Optional[float] = None,
    ) -> None:
        """
        Placeholder: update belief for this auctioned item using observed price.

        Inputs you *might* want:
          - item_id: which item was auctioned
          - price_paid: observed second price
          - winning_team: which team won (optional)
          - my_bid: what we bid (optional; helps interpret whether price might equal our bid)

        For now, this function is intentionally empty.
        We'll later implement something like:
            P(T | v, price) ∝ P(T | v) * likelihood(price | T)
        where likelihood(price | T) is derived from an opponent-bidding model.
        """
        # TODO: implement Bayesian update using a likelihood model for prices.
        return
