What should we do: 
    beliefs: 
        * update item valuation after price reveal. 
        * update rest of the items after prior change.
        ~~~ lets say after price reveal we're 0.8 sure that it is mixed, how do we model our priors based on our beliefs? 


    Competition modeling: 
        * we need to keep track of which team has what budget. 

    strategy: 
    given beliefs we need some kind state machine (remaining rounds...): 
        * competition strategy: 
            - which item class do we target? 
            - what are our options for screwing other teams? 
        * choose current round strategy: 
            - should we just do truthful? 
            - should we make sure we're winning this round? 
        * budget pacing (what are the funds we give this round)

HIGH: 
    v = 18, 
    2nd highest = 17
    utility = 1 

LOW: 
    v = 9, 
    2nd highest = 7.5
    utility = 1.5 

MIXED: 
    v = 18 
    2nd highest = 15 
    utility = 3


Observation: 
    * with mixed distribution - we have high chance for greater utility at the expense of higher price. 
    * high value items are strictly the worse. 
    * our confidence level of which category each item belong to, become much more clear in later rounds. 
    * meaning decisions made based on our beliefs should be left for later rounds. 
    * if we're confident that a certain item is for example ALL.
    and our value is pretty low in this distribution, what can we do? 
    we can overbid, and make sure the winner utility deminishes. at the risk of negative utility. 

Research Questions: 
    * how do beliefs update regulary change? when are we confident about some items? 
    * do we have a target utility we want acheive? we need some benchmarks? 
    An example 


Simple competition strategies: 
    

CompetitorBudget (
    History
)

BID(
    Number Round,
    Current Util,
    Item,
    Budget, 
    Beliefs, 
    CompetitorBudget
) -> R

First strategy: 

We're determinimg how each parameter affects the mukplicity factor. 
v * round_shade(Number_round) * budget_shade(Budget) * belief_shade(Beliefs) * competitor_shade(CompetitorBudget)

return Value() * Shading()


Second Algorithm: 
Meaningful functions, such as
 stress factor -> late round, high budget, low util -> increase my bid (bluffing to screw others)


Lets define shade -> should return a multplicity factor to the bid. 
shade: XXX -> [0, 1]
Strategy set is subset of available shades, marked S 
v * Pi_{shade in S}({(1 + shade_i(args))})

main task until next meeting, think of at least 5 shade functions. 
