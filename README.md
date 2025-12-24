hiiiii this is yaniv
this is from vscode
# 🎯 AGT Auto-Bidding Competition - Student Package

Welcome to the AGT 2025-2026 Auto-Bidding Challenge!

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Read the Documentation
- **[STUDENT_GUIDE.md](STUDENT_GUIDE.md)** - Complete implementation guide (START HERE!)
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - Quick reference for common tasks

### 3. Create Your Agent
```bash
# Create your team folder
mkdir teams/my_team

# Copy the template
cp AGENT_TEMPLATE.py teams/my_team/bidding_agent.py

# Edit teams/my_team/bidding_agent.py to implement your strategy
```

### 4. Test Your Agent
```bash
# Validate your agent
python main.py --mode validate --validate teams/my_team/bidding_agent.py

# Test against example agents (10 games)
python simulator.py --your-agent teams/my_team/bidding_agent.py --num-games 10
```

### 5. Register Your Team
Your team name (folder name) and student IDs must be registered before submission. Contact the course staff with:
- Your chosen team name (must match your folder name exactly)
- List of all team member student IDs

---

## 📋 Competition Overview

**Objective**: Maximize utility in sequential second-price auctions

- **Items**: 20 total per game (15 auctioned each game)
- **Budget**: 60 units per game
- **Rounds**: 15 auction rounds per game
- **Games**: 5 games per stage
- **Auction Type**: Second-price sealed-bid (Vickrey)
- **Timeout**: 3 seconds per bid
- **Bid Precision**: Rounded to 2 decimal places

**Scoring**: `Utility = Σ(Values Won) - Σ(Prices Paid)`

---

## 📁 Package Structure

```
AGT_Competition_Package/
├── STUDENT_GUIDE.md          # Complete implementation guide (includes registration)
├── QUICK_REFERENCE.md         # Quick reference
├── AGENT_TEMPLATE.py          # Starter template
├── simulator.py               # Test your agent locally
├── main.py                    # Competition system
├── validate_registration.py   # Validate team registration
├── examples/                  # Reference strategies
│   ├── truthful_bidder.py
│   ├── budget_aware_bidder.py
│   ├── strategic_bidder.py
│   └── random_bidder.py
├── teams/                     # Your workspace
│   └── team_registration.json # Team-to-student mapping
└── src/                       # Competition system code
```

---

## 💡 Examples

Check the `examples/` folder for reference implementations:
- **truthful_bidder.py** - Bids true valuation
- **budget_aware_bidder.py** - Budget-conscious strategy
- **strategic_bidder.py** - Opponent modeling strategy
- **random_bidder.py** - Random baseline

---

## 🆘 Need Help?

1. Check [STUDENT_GUIDE.md](STUDENT_GUIDE.md) for detailed explanations
2. Use [QUICK_REFERENCE.md](QUICK_REFERENCE.md) for quick lookups
3. Test with `--verbose` flag to see detailed execution
4. Ask on the course forum

---

## 🏆 Good Luck!

May the best strategy win! 🎯
