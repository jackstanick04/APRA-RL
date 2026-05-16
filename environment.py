import numpy as np
# parent class
import gymnasium as gym
# used for action/observation data checking
from gymnasium import spaces

# env class, inherits from gym
class Apra_env(gym.Env):
    
    # constructor method, inheriting environment
    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, signal_noise, valuation_weight, obs_size, render_mode = None):
        super().__init__()

        # auction-env specific variables, as well as the agent signals and reward tracker
        self.num_rounds = num_rounds
        self.current_round = 0
        self.num_opponents = num_opponents
        self.reserve_price = reserve_price
        self.reimbursement_rates = reimbursement_rates
        self.bid_cost = bid_cost
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_signal = 0.0
        self.agent_max_bid_holder = False
        self.opp_signals = np.zeros(self.num_opponents)
        self.signal_noise = signal_noise
        self.valuation_weight = valuation_weight
        self.obs_size = obs_size
        self.render_mode = render_mode

        # define action and observation spaces
            # all already 0, 1 except round # which we normalize
            # Box: all valid cartesian products
            # action: bid // observation: agent_signal, max_bid, agent_win_loss, round_num
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32)
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (self.obs_size,), dtype = np.float32)


    # reset method to start from scratch
    def reset(self, seed = None, options = None):

        # reset auction-specific variables
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False

        # seed for rng
        super().reset(seed = seed)

        # make a base value, then add some noise to each person (keep between 0 and 1)
        true_value = self.np_random.uniform(0.0, 1.0)
        self.opp_signals = np.clip(true_value + self.np_random.standard_normal(self.num_opponents) * self.signal_noise, 0.0, 1.0)

        # check if special fixed signal for agent (for debugging)
        if options and options.get("fixed_signal") is not None:
            self.agent_signal = options["fixed_signal"]
        else: 
            self.agent_signal = np.clip(true_value + self.np_random.standard_normal() * self.signal_noise, 0.0, 1.0)

        # make observation space to give to the agent
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), 0.0], dtype = np.float32)

        # extra info for debugging
        info = {
            "agent_signal": self.agent_signal,
            "opp_signals": self.opp_signals
        }

        return observation, info

    # step function, running the logic for a round of bidding
    def step(self, action):
        
        # ensure not abstaining
        agent_bid = None if action[0] < 0.01 else action[0]

        # find max bid out of everyone and store index
        opp_bids = self.opp_bids()
        all_bids = [agent_bid] + opp_bids
        round_max_bid = max((bid for bid in all_bids if bid is not None), default = 0.0)
        # if everyone asbtains all bids are 0, so nobody should have the max bid or get reimbursed (stay -1)
        round_max_bid_index = all_bids.index(round_max_bid) if round_max_bid > 0.0 else -1

        # find all valuations (again, agent is index 0)
        all_signals = [self.agent_signal] + list(self.opp_signals)
        all_vals = [self.valuation(all_signals, i, self.max_bid) for i in range(self.num_opponents + 1)]

        # ensure it is greater than last round, and find reimbursement
        # also check if agent is leading
        reimbursements = [0] * (self.num_opponents + 1)
        if round_max_bid > self.max_bid:
            reimbursements[round_max_bid_index] = (round_max_bid - self.max_bid) * self.reimbursement_rates[self.current_round]
            self.max_bid = round_max_bid
            self.max_bid_index = round_max_bid_index
            self.agent_max_bid_holder = self.max_bid_index == 0

        # if last round, check if bid is over reserve price for allocation
        last_round = self.current_round + 1 == self.num_rounds
        won = last_round and (self.max_bid >= self.reserve_price)
        
        # calculate rewards for all (maybe only need agent)
        rewards = [0] * (self.num_opponents + 1)
        for bidder in range(self.num_opponents + 1):
            rewards [bidder] = self.reward(all_bids[bidder], reimbursements[bidder], last_round, won, all_vals[bidder])

        # create the observation space (standardize round number to [0, 1] first)
        std_round = self.current_round / self.num_rounds
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), std_round], dtype = np.float32)

        # game updates and info dict (can always add more)
        self.current_round += 1
        extra_info = {
            "Other Agent Rewards" : rewards[1:]
        }
        
        # return the observation space, reward, if last round, truncated = False, and auxillary info 
        return observation, rewards[0], last_round, False, extra_info

    # reward function based on round number and win status
    # valuation is determined in agent class--assumed to be a constant for now
    def reward(self, bid, reimbursement, last_round, won, valuation):
        
        # check if they won the round or not, and if not, check if they abstained
        if reimbursement == 0:
            if bid is None:
                return 0
            else: 
                return -(self.bid_cost)
            
        # if won, update utility if it is the last round
        else:
            if last_round and won: 
                return valuation - bid + reimbursement - self.bid_cost
            else:
                return reimbursement - self.bid_cost
            
    # opponent bid function
    # NEEDS TO BE UPDATED BIG TIME DOWNT THE ROAD
    def opp_bids(self): 

        # check if they are already leading or they are not bidding large enough
        opp_bids = []
        for i in range(self.num_opponents):

            bid = round(self.opp_signals[i] * 0.75, 2)
            leader = self.max_bid_index == i + 1 # max bid index includes the agent at 0

            if bid > self.max_bid and not leader:
                opp_bids.append(bid)
            else: 
                opp_bids.append(None)

        return opp_bids

    # valuation (for agent and opponents) function based on max bid
    # this function may be updated to be more complex!!
    def valuation(self, signals, bidder_num, stat): 
        # no randomness because we want our value function to be deterministic--if it's not, what's the point of a signal?
        return np.clip(signals[bidder_num] + (stat * self.valuation_weight), 0.0, 1.0)

    # render basic text to the terminal for tracking when we choose to
    def render(self):
        if self.render_mode == "human":
            print(f"Round Number: {int(self.current_round)}")
            print(f"Agent Signal: {self.agent_signal}")
            print(f"Opponent Signals: {self.opp_signals}")
            print(f"Highest Bid: {self.max_bid}")
        


    








        

        


