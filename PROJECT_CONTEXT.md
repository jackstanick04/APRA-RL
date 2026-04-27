PROJECT OVERVIEW
    - Developing Reinforcement Learning agents in the context of an Ascending Price Reimbursement Auction to model optimal bidding behavior, in order to maximize revenue for the auctioneer

ENVIRONMENT
    - will implement gymnasium
        - like a gaming console, in that it handles all the game updates and backend math, and the player just interacts with the environment by acting and seeing its result
    - 5 major methods to implement, as well as other relevant methods and attributes
        - constructor, reset, step, close, and render (not as important)
        - helper methods for the step function (like calculate reward)
        - Step method
            - takes in bids and valuations from all
            - finds max bid, and if they raised it, finds reimbursement
            - finds reward value for all players (based on if last round or not)
            - returns agent's reward, the observation, end of auction, and auxillary reward info of others

NEXT STEP: start the agent/main class?
    - get help from francisco here


