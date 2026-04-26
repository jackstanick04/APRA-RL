PROJECT OVERVIEW
    - Developing Reinforcement Learning agents in the context of an Ascending Price Reimbursement Auction to model optimal bidding behavior, in order to maximize revenue for the auctioneer

ENVIRONMENT
    - will implement gymnasium
        - like a gaming console, in that it handles all the game updates and backend math, and the player just interacts with the environment by acting and seeing its result
    - 5 major methods to implement, as well as other relevant methods and attributes
        - constructor, reset, step, close, and render (not as important)
    - i am down to the step function and here are the notes: 
        Step method
            Parameters: agent’s bid, list of bids from other players
            Returns: reward for agent, list of rewards for other players, boolean if auction over or not
            Flow
                Check if there is a new max bid, and keep index
                    Abstention uses None
                    Calculate the reimbursement for that agent
                Calculate rewards for all
                    Check if last round in here
                Update round number
            Reward function:
                Parameters: abstention, reimbursement, last round, won item, valuation?
                Returns: reward
                If last round: 
                    Also include the valuation for the item
                If not: 
                    Reimbursement - constant costas 
                    If none: return 0

    NEED TO FINISH UP STEP FUNCTION (correct return values and stuff)
    THEN GO INTO VALUATIONS AND AGENT CLASS?
