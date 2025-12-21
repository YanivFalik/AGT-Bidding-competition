# AGT Competition - Student Implementation Guide

## 📋 Table of Contents

1. [Getting Started](#getting-started)
2. [Understanding the Competition](#understanding-the-competition)
3. [Agent Interface Overview](#agent-interface-overview)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Testing Your Agent](#testing-your-agent)
6. [Strategy Development Tips](#strategy-development-tips)
7. [Common Pitfalls](#common-pitfalls)
8. [Debugging Guide](#debugging-guide)
9. [Team Registration](#team-registration)
10. [Submission Guidelines](#submission-guidelines)

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11 or higher
- Basic understanding of game theory and auctions
- Familiarity with Python programming

### Installation

1. Clone or download the competition repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python main.py --mode validate --validate examples/truthful_bidder.py
```

You should see: `✓ Agent validation PASSED`

---

## 🎯 Understanding the Competition

### Educational Objectives

This competition will test your understanding of:

1. **Auction Theory** - Understanding second-price auction properties and strategic implications
2. **Algorithmic Strategy** - Designing computational approaches to sequential decision-making
3. **Budget Optimization** - Resource allocation under scarcity and uncertainty
4. **Competitive Analysis** - Reasoning about opponent behavior and market dynamics
5. **Information Economics** - Decision-making with partial information and limited market signals

Key concepts to review:
- **Truthful Bidding**: When is bidding your true value optimal in second-price auctions?
- **Budget Constraints**: How do financial limitations change bidding strategies?
- **Sequential Games**: Planning and adaptation in multi-round environments
- **Information Asymmetry**: Using private valuations strategically
- **Information Revelation**: Making inferences about others' behavior

### Competition Structure

```
Stage 1: Qualification Round
├── All teams divided into 5-team arenas
├── Each arena plays 5 games
├── Top scorer from each arena advances to Stage 2
│
Stage 2: Championship Round
├── All qualified teams in single arena
├── Play 5 games
└── Final rankings determine winners
```

### Single Game Flow

```
Pre-Game: You receive
├── Your valuation vector (20 items with values)
├── Initial budget: 60 units
├── Opponent team IDs
└── Unknown: Which 15 items will be auctioned (learned online)
└── Unknown: The order items will be auctioned

During Game: For each of 15 rounds
├── System announces which item is being auctioned (online revelation)
├── You submit your bid (you have 3 seconds max)
├── Highest bidder wins, pays second-highest bid
└── All agents learn: winner identity + price paid

Post-Game: Calculate utility
└── Utility = Σ(values of items won) - Σ(prices paid)
```

### Key Rules

- **Auction Type**: Second-price sealed-bid (Vickrey auction)
- **Budget**: 60 units per game (does NOT carry over between games)
- **Timeout**: 3 seconds per bid decision
- **Bid Precision**: All bids rounded to 2 decimal places
- **Information**: After each round, you learn winner + price (NOT all bids)
- **Scoring**: Total utility across all games in your stage

### Item Valuation Distribution

Each game, valuations are generated as follows:

- **6 items**: High-value for ALL teams (each team gets value U[10, 20])
- **4 items**: Low-value for ALL teams (each team gets value U[1, 10])
- **10 items**: Mixed values (each team gets value U[1, 20])

**Important**: You don't know which category each item belongs to! A high-value item in your valuation might be high-value for everyone (competitive) or only for you (less competitive).

---

## 🔧 Agent Interface Overview

### Required Class Structure

Every team must implement a class called `BiddingAgent` with these methods:

```python
class BiddingAgent:
    def __init__(self, team_id: str, valuation_vector: dict, 
                 budget: float, opponent_teams: list):
        """
        Initialize your agent at the start of each game.
        
        Args:
            team_id: Your unique team identifier (UUID)
            valuation_vector: Dict mapping item_id to your valuation
                Example: {"item_0": 15.3, "item_1": 8.2, ..., "item_19": 12.7}
            budget: Initial budget (always 60)
            opponent_teams: List of opponent team IDs in the same arena
                Example: ["Team_A", "Team_B", "Team_C", "Team_D"]
                Use this to track and model each opponent separately
        """
        pass
    
    def bidding_function(self, item_id: str) -> float:
        """
        REQUIRED: Return your bid for the current item.
        This is called once per round.
        
        Args:
            item_id: The item being auctioned (e.g., "item_7")
        
        Returns:
            float: Your bid amount
                - Must be >= 0
                - Should be <= your current budget
                - Bids over budget are automatically capped
                - Bids are automatically rounded to 2 decimal places
        
        Important:
            - You have 3 seconds maximum to return
            - Timeout or error = bid of 0
            - This is a SECOND-PRICE auction: winner pays second-highest bid
            - All bids are rounded to 2 decimal places (e.g., 15.678 → 15.68)
        """
        pass
    
    def update_after_each_round(self, item_id: str, winning_team: str, 
                                price_paid: float):
        """
        REQUIRED: Called after each auction round with public information.
        Use this to update your strategy, beliefs, or opponent models.
        
        Args:
            item_id: The item that was just auctioned
            winning_team: Team ID of the winner (empty string if no winner)
            price_paid: Price the winner paid (second-highest bid)
        
        Note:
            - You do NOT see all bids, only winner and price
            - Your budget is automatically updated by the system
            - Use this to learn about opponent behavior
        """
        pass
```

### Provided Attributes (Auto-managed)

These attributes are automatically maintained by the base class:

```python
self.team_id              # Your team identifier
self.valuation_vector     # Your valuations for all 20 items
self.budget               # Current remaining budget (updated automatically)
self.initial_budget       # Starting budget (60)
self.opponent_teams       # List of opponent team IDs in your arena
self.utility              # Current utility (updated automatically)
self.items_won            # List of items you've won (updated automatically)
```

---

## 📝 Step-by-Step Implementation

### Step 1: Create Your Team Directory

```bash
mkdir teams/your_team_name
cd teams/your_team_name
```

### Step 2: Copy the Template

Create `bidding_agent.py`:

```python
"""
Team: [Your Team Name]
Members: [Student 1, Student 2, Student 3]
Strategy: [Brief description]
"""

from typing import Dict, List


class BiddingAgent:
    def __init__(self, team_id: str, valuation_vector: Dict[str, float], 
                 budget: float, opponent_teams: List[str]):
        # Required attributes
        self.team_id = team_id
        self.valuation_vector = valuation_vector
        self.budget = budget
        self.initial_budget = budget
        self.opponent_teams = opponent_teams
        self.utility = 0
        self.items_won = []
        
        # Your custom attributes here
        self.rounds_completed = 0
        self.total_rounds = 15  # Always 15 rounds per game
        # Example: self.opponent_bids = {opp: [] for opp in opponent_teams}
        # Example: self.beliefs = {opp: {} for opp in opponent_teams}
        
    def _update_available_budget(self, item_id: str, winning_team: str, 
                                 price_paid: float):
        """DO NOT MODIFY - Managed by system"""
        if winning_team == self.team_id:
            self.budget -= price_paid
            self.items_won.append(item_id)
    
    def update_after_each_round(self, item_id: str, winning_team: str, 
                                price_paid: float):
        """Update your strategy after each round"""
        # System update (DO NOT REMOVE)
        self._update_available_budget(item_id, winning_team, price_paid)
        if winning_team == self.team_id:
            self.utility += (self.valuation_vector[item_id] - price_paid)
        
        # YOUR CODE HERE: Update beliefs, track opponent behavior, etc.
        self.rounds_completed += 1
        
        return True
    
    def bidding_function(self, item_id: str) -> float:
        """
        YOUR MAIN STRATEGY IMPLEMENTATION
        
        Return your bid for the current item.
        """
        # Get your valuation for this item
        my_valuation = self.valuation_vector.get(item_id, 0)
        
        # IMPLEMENT YOUR STRATEGY HERE
        bid = my_valuation  # Simple truthful bidding (IMPROVE THIS!)
        
        # Ensure bid is valid
        bid = max(0, min(bid, self.budget))
        
        return bid
```

### Step 3: Implement Your Strategy

Replace the simple `bidding_function` with your strategy. Consider:

1. **Budget Management**: How much to spend now vs. save for later?
2. **Opponent Modeling**: Can you learn from observed prices?
3. **Value Assessment**: Which items are worth competing for?
4. **Risk Management**: When to bid aggressively vs. conservatively?

### Step 4: Test Your Agent

```bash
# Validate your agent
python main.py --mode validate --validate teams/your_team_name/bidding_agent.py

# Test against example agents
python simulator.py --your-agent teams/your_team_name/bidding_agent.py --num-games 10
```

---

## 🧪 Testing Your Agent

### Local Validation

Verify your agent implements the interface correctly:

```bash
python main.py --mode validate --validate teams/your_team_name/bidding_agent.py
```

Expected output:
```
✓ Test bid successful: X.XX (took 0.XXXs)
✓ Agent validation PASSED
```

### Simulator Testing

Use the provided simulator to test against example strategies:

```bash
# Test against all example agents (10 games)
python simulator.py --your-agent teams/your_team_name/bidding_agent.py --num-games 10

# Test against specific opponent
python simulator.py --your-agent teams/your_team_name/bidding_agent.py \
                    --opponent examples/strategic_bidder.py --num-games 5

# Verbose output to see detailed game flow
python simulator.py --your-agent teams/your_team_name/bidding_agent.py \
                    --num-games 3 --verbose

# Test with different random seeds for consistency
python simulator.py --your-agent teams/my_team/bidding_agent.py \
                    --num-games 10 --seed 6431
```

### Analyze Your Performance

The simulator generates a report showing:
- Win rate against each opponent
- Average utility per game
- Budget utilization
- Items won statistics
- Execution time analysis

---

## 💡 Strategy Development Tips

### Understanding Second-Price Auctions

In a standard second-price auction (without budget constraints):
- **Optimal Strategy**: Bid your true valuation
- **Why**: You win if your value > others, pay less than your value
- **With Budget Constraints**: This changes! You need to be more strategic

### Key Strategic Considerations

#### 1. Budget Pacing

```python
# Calculate how much budget per remaining round
total_rounds = 15  # Always 15 rounds per game
rounds_remaining = total_rounds - self.rounds_completed
budget_per_round = self.budget / rounds_remaining if rounds_remaining > 0 else 0

# Be more aggressive as game progresses
progress = self.rounds_completed / total_rounds
aggressiveness = 0.7 + (0.3 * progress)  # 70% to 100%
```

#### 2. Value Classification

```python
# Classify items based on your valuation
high_value_threshold = 12.0
low_value_threshold = 8.0

if my_valuation > high_value_threshold:
    # Might be competitive, bid carefully
    bid = my_valuation * 0.9
elif my_valuation < low_value_threshold:
    # Low value, bid conservatively
    bid = my_valuation * 0.5
else:
    # Medium value
    bid = my_valuation * 0.7
```

#### 3. Opponent Modeling

```python
# Track observed prices
self.observed_prices.append(price_paid)

# Estimate market competitiveness
avg_price = sum(self.observed_prices) / len(self.observed_prices)
max_price = max(self.observed_prices)

# Adjust strategy based on market
if avg_price > 10:  # Competitive market
    # Be more conservative on medium-value items
    pass
```

#### 4. Information Revelation

What you can learn from each round:
- **Price paid**: Indicates second-highest bid (at least)
- **Winner identity**: Track which teams are winning
- **No winner**: Everyone bid low (item not valuable to others)

```python
# Track opponent success
if winning_team != self.team_id and winning_team:
    self.opponent_wins[winning_team] = \
        self.opponent_wins.get(winning_team, 0) + 1
```

### Advanced Strategies

#### Bayesian Belief Updates

```python
# Maintain beliefs about opponent valuations
# Update after observing their bids (revealed when they win)
# Use to predict future competition
```

#### Shading Strategy

```python
# In second-price auctions with budgets, you might want to "shade" bids
# Bid less than true value to preserve budget
shading_factor = 0.8  # Bid 80% of valuation
bid = my_valuation * shading_factor
```

#### End-Game Strategy

```python
# Near end of game, spend remaining budget more aggressively
if rounds_remaining <= 3:
    # Less need to preserve budget
    bid = min(my_valuation, self.budget * 0.8)
```

---

## ⚠️ Common Pitfalls

### 1. Timeout Errors

**Problem**: Your `bidding_function` takes > 3 seconds
```python
# BAD: Complex computation that might timeout
def bidding_function(self, item_id):
    # Huge nested loops...
    for i in range(1000000):
        for j in range(1000000):
            # complex calculation
    return bid
```

**Solution**: Keep computation simple and fast
```python
# GOOD: Pre-compute what you can in __init__ or update_after_each_round
def bidding_function(self, item_id):
    # Quick lookup or simple calculation
    return self.precomputed_bids.get(item_id, 0)
```

### 2. Budget Exhaustion

**Problem**: Spending all budget early, nothing left for later rounds
```python
# BAD: Always bidding full valuation
def bidding_function(self, item_id):
    return self.valuation_vector[item_id]  # Might run out of budget!
```

**Solution**: Pace your budget
```python
# GOOD: Reserve budget for future rounds
def bidding_function(self, item_id):
    rounds_left = 15 - self.rounds_completed
    max_bid_now = self.budget / max(1, rounds_left) * 1.5
    return min(self.valuation_vector[item_id], max_bid_now)
```

### 3. Not Updating State

**Problem**: Ignoring information from previous rounds
```python
# BAD: Not learning from observations
def update_after_each_round(self, item_id, winning_team, price_paid):
    self._update_available_budget(item_id, winning_team, price_paid)
    # Not tracking anything!
    return True
```

**Solution**: Learn from each round
```python
# GOOD: Track patterns and update beliefs
def update_after_each_round(self, item_id, winning_team, price_paid):
    self._update_available_budget(item_id, winning_team, price_paid)
    
    # Learn from observations
    self.price_history.append(price_paid)
    self.avg_price = sum(self.price_history) / len(self.price_history)
    
    return True
```

### 4. Invalid Return Values

**Problem**: Returning wrong type or invalid bid
```python
# BAD: Various errors
def bidding_function(self, item_id):
    return None  # Error!
    return -5    # Negative bid!
    return "10"  # String instead of float!
```

**Solution**: Always return valid float
```python
# GOOD: Ensure valid return
def bidding_function(self, item_id):
    bid = self.calculate_bid(item_id)
    return max(0.0, float(bid))  # Ensure non-negative float
```

### 5. Forgetting Budget Constraints

**Problem**: Not checking if you can afford your bid
```python
# BAD: Bidding more than available budget
def bidding_function(self, item_id):
    return 100  # But budget might be only 10!
```

**Solution**: System caps bids automatically, but you should check:
```python
# GOOD: Respect budget
def bidding_function(self, item_id):
    desired_bid = self.calculate_bid(item_id)
    return min(desired_bid, self.budget)
```

---

## 🐛 Debugging Guide

### Enable Verbose Logging

Test with verbose mode to see all decisions:

```bash
python simulator.py --your-agent teams/your_team_name/bidding_agent.py \
                    --num-games 1 --verbose
```

### Add Print Statements

```python
def bidding_function(self, item_id):
    my_val = self.valuation_vector[item_id]
    print(f"[{self.team_id}] Round {self.rounds_completed}: "
          f"Item {item_id}, Value={my_val:.2f}, Budget={self.budget:.2f}")
    
    bid = self.calculate_bid(item_id)
    print(f"[{self.team_id}] Bidding: {bid:.2f}")
    
    return bid
```

### Check Execution Time

```python
import time

def bidding_function(self, item_id):
    start = time.time()
    
    # Your strategy
    bid = self.calculate_bid(item_id)
    
    elapsed = time.time() - start
    if elapsed > 1.5:  # Warning if taking too long
        print(f"WARNING: Bid took {elapsed:.3f}s")
    
    return bid
```

### Test Edge Cases

```python
# Test with minimal budget
# Test with first/last rounds
# Test with all high-value items
# Test with all low-value items
```

---

## 📤 Submission Guidelines

### File Structure

Your submission should be organized as:

```
your_team_name/
└── bidding_agent.py
```

### Checklist Before Submission

- [ ] File named exactly `bidding_agent.py`
- [ ] Class named exactly `BiddingAgent`
- [ ] All required methods implemented
- [ ] Validation passes: `python main.py --mode validate --validate your_agent.py`
- [ ] No external dependencies beyond numpy, scipy, standard library
- [ ] No file I/O, network access, or system calls
- [ ] Agent runs in < 3 seconds per bid
- [ ] Code is well-commented
- [ ] Team member names in header comment

### What to Submit

1. Your `bidding_agent.py` file
2. (Optional) A brief strategy document (1-2 pages) explaining your approach

### Submission Deadline

[TBD - Check course website]

### Testing Recommendations

Before submitting:
1. Test against all example agents (at least 50 games each)
2. Test with different random seeds
3. Verify consistent performance (not just lucky wins)
4. Check average utility is positive
5. Ensure no timeout errors

---

## 🎓 Example: Simple Strategic Agent

Here's a complete example showing basic strategic concepts:

```python
"""
Example Strategic Agent
Demonstrates: Budget pacing, opponent modeling, adaptive bidding
"""

from typing import Dict, List


class BiddingAgent:
    def __init__(self, team_id: str, valuation_vector: Dict[str, float], 
                 budget: float, opponent_teams: List[str]):
        self.team_id = team_id
        self.valuation_vector = valuation_vector
        self.budget = budget
        self.initial_budget = budget
        self.opponent_teams = opponent_teams
        self.utility = 0
        self.items_won = []
        
        # Strategy state
        self.rounds_completed = 0
        self.total_rounds = 15
        self.price_history = []
        self.opponent_wins = {opp: 0 for opp in opponent_teams}
        
        # Pre-compute some statistics
        self.my_avg_valuation = sum(valuation_vector.values()) / len(valuation_vector)
        self.my_max_valuation = max(valuation_vector.values())
    
    def _update_available_budget(self, item_id: str, winning_team: str, 
                                 price_paid: float):
        if winning_team == self.team_id:
            self.budget -= price_paid
            self.items_won.append(item_id)
    
    def update_after_each_round(self, item_id: str, winning_team: str, 
                                price_paid: float):
        self._update_available_budget(item_id, winning_team, price_paid)
        
        if winning_team == self.team_id:
            self.utility += (self.valuation_vector[item_id] - price_paid)
        
        # Learn from observations
        if price_paid > 0:
            self.price_history.append(price_paid)
        
        if winning_team and winning_team != self.team_id:
            self.opponent_wins[winning_team] = \
                self.opponent_wins.get(winning_team, 0) + 1
        
        self.rounds_completed += 1
        return True
    
    def bidding_function(self, item_id: str) -> float:
        my_valuation = self.valuation_vector.get(item_id, 0)
        
        # Calculate market conditions
        if self.price_history:
            avg_market_price = sum(self.price_history) / len(self.price_history)
            max_market_price = max(self.price_history)
        else:
            avg_market_price = 8.0  # Initial estimate
            max_market_price = 15.0
        
        # Calculate rounds remaining
        rounds_remaining = self.total_rounds - self.rounds_completed
        
        if rounds_remaining == 0:
            return 0
        
        # Budget pacing: how much can we spend per round?
        budget_per_round = self.budget / rounds_remaining
        
        # Classify item value relative to market
        if my_valuation > max_market_price:
            # High-value item - bid aggressively but save some budget
            bid_fraction = 0.85
        elif my_valuation > avg_market_price:
            # Medium-value item
            bid_fraction = 0.70
        else:
            # Low-value item - bid conservatively
            bid_fraction = 0.50
        
        # Calculate base bid
        bid = my_valuation * bid_fraction
        
        # Adjust for game progress (more aggressive near end)
        progress = self.rounds_completed / total_rounds
        if progress > 0.7:  # Last 30% of game
            bid = bid * 1.1  # Bid 10% more aggressively
        
        # Ensure we don't exceed budget
        bid = min(bid, self.budget)
        
        # Don't overspend early unless it's very valuable
        if rounds_remaining > 5 and bid > budget_per_round * 2:
            bid = budget_per_round * 2
        
        return max(0, bid)
```

---

## 🎫 Team Registration

### Overview

Before submitting your code, your team must be registered with the course staff. This maps your student IDs to your team name for grading and accountability.

### Registration Requirements

Provide the following to the course staff:

1. **Team Name** - Must match your submission folder name exactly (case-sensitive)
2. **Team Members** - List of all student IDs

### Example Registration

If your team is named `team_alpha` with three members:
- Team folder: `teams/team_alpha/`
- Student IDs: `["123456789", "987654321", "555555555"]`

### Important Notes

- ✅ Team name must exactly match your folder name
- ✅ All team members must be registered before submission deadline
- ❌ Late registrations may not be accepted

### Validation (For Course Staff)

Course staff can validate the registration file using:

```bash
python validate_registration.py
```

This checks:
- JSON syntax validity
- No duplicate team names
- No duplicate student IDs across teams
- All team folders match registration entries

---

## 📤 Submission Guidelines

### Submission Checklist

Before submitting, ensure:

- [ ] Team is registered with course staff (team name + student IDs)
- [ ] Agent passes validation: `python main.py --mode validate --validate teams/your_team/bidding_agent.py`
- [ ] Agent tested with simulator: `python simulator.py --your-agent teams/your_team/bidding_agent.py`
- [ ] No syntax errors or import issues
- [ ] Bidding function completes within 3 seconds
- [ ] Agent handles all edge cases (zero budget, final round, etc.)
- [ ] Code is properly commented and readable
- [ ] Team folder structure is correct:
  ```
  teams/your_team_name/
      bidding_agent.py
  ```

### File Structure

Your submission must follow this exact structure:

```python
class BiddingAgent:
    def __init__(self):
        # Your initialization code
        pass
    
    def on_auction_start(self, total_rounds: int, total_items: int, 
                        initial_budget: float) -> bool:
        # Initialize game state
        return True
    
    def on_new_round(self, round_number: int) -> bool:
        # Prepare for new round
        return True
    
    def on_valuation_received(self, item_id: str, valuation: float) -> bool:
        # Store valuation
        return True
    
    def bidding_function(self, item_id: str) -> float:
        # Return your bid (must complete in 3 seconds)
        return bid
    
    def on_auction_result(self, item_id: str, winner_team_id: str, 
                         winner_bid: float, price: float) -> bool:
        # Update state with auction outcome
        return True
```

### Common Submission Errors

1. **Wrong folder structure** - Must be `teams/team_name/bidding_agent.py`
2. **Missing methods** - All required methods must be implemented
3. **Class name mismatch** - Class must be named `BiddingAgent`
4. **Timeout issues** - Bidding function exceeds 3 seconds
5. **Unhandled exceptions** - Agent crashes on edge cases
6. **Team name mismatch** - Folder name doesn't match registration

### Testing Before Submission

```bash
# 1. Validate syntax and structure
python main.py --mode validate --validate teams/your_team/bidding_agent.py

# 2. Test against examples (quick test)
python simulator.py --your-agent teams/your_team/bidding_agent.py --num-games 5

# 3. Extensive testing (recommended)
python simulator.py --your-agent teams/your_team/bidding_agent.py --num-games 50

# 4. Test with verbose output for debugging
python simulator.py --your-agent teams/your_team/bidding_agent.py --num-games 5 --verbose
```

---

## 📚 Additional Resources

- **QUICK_REFERENCE.md** - Quick reference for common tasks
- **Example Agents** - Check `examples/` folder for different strategies
- **validate_registration.py** - Validation tool for course staff
- **Course Forum** - Post questions and discuss strategies

---

## 🏆 Good Luck!

### Remember:

- ✅ Start simple, iterate and improve
- ✅ Test extensively before submitting
- ✅ Register your team early
- ✅ Budget management is crucial
- ✅ Learn from opponent behavior
- ✅ Handle edge cases properly
- ✅ Have fun and be creative!

May the best strategy win! 🎯
