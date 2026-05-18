import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Apra_env(gym.Env):
    
    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, signal_noise, valuation_weight, obs_size, render_mode = None):
        super().__init__()

        # auction parameters (fixed)
        self.num_rounds = num_rounds
        self.num_opponents = num_opponents
        self.reserve_price = reserve_price
        self.reimbursement_rates = reimbursement_rates
        self.bid_cost = bid_cost
        self.signal_noise = signal_noise
        self.valuation_weight = valuation_weight
        self.obs_size = obs_size
        self.render_mode = render_mode

        # auction states (reset per episode)
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False
        self.agent_signal = 0.0
        self.opp_signals = np.zeros(self.num_opponents)
        self.opponents = [] # for debugging

        # Box spaces ensure that all observation and action values are in [0,1] range
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32) # action: bid
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (self.obs_size,), dtype = np.float32) # observation: agent_signal, max_bid, agent_max_bid_holder, round_num (gets scaled to [0,1])

    def reset(self, seed = None, options = None):

        # only auction state variables reset
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False
        super().reset(seed = seed) # seed for agent signal debugging

        # make a base value, then add some noise to each person (keep between 0 and 1)
        true_value = self.np_random.uniform(0.0, 1.0)
        self.opp_signals = np.clip(true_value + self.np_random.standard_normal(self.num_opponents) * self.signal_noise, 0.0, 1.0)
        # check if special fixed signal for agent (for debugging)
        if options and options.get("fixed_signal") is not None:
            self.agent_signal = options["fixed_signal"]
        else: 
            self.agent_signal = np.clip(true_value + self.np_random.standard_normal() * self.signal_noise, 0.0, 1.0)

        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), 0.0], dtype = np.float32)

        # extra info for debugging
        info = {
            "agent_signal": self.agent_signal,
            "opp_signals": self.opp_signals
        }

        return observation, info

    def step(self, action):
        
        agent_bid = None if action[0] < 0.01 else action[0] # abstention

        opp_bids = self.opp_bids()
        all_bids = [agent_bid] + opp_bids
        round_max_bid = max((bid for bid in all_bids if bid is not None), default = 0.0)
        round_max_bid_index = all_bids.index(round_max_bid) if round_max_bid > 0.0 else -1 # if everyone asbtains, nobody gets reimbursed

        # agent is index 0
        all_signals = [self.agent_signal] + list(self.opp_signals)
        all_vals = [self.valuation(all_signals, i, self.max_bid) for i in range(self.num_opponents + 1)]

        reimbursements = [0] * (self.num_opponents + 1)
        if round_max_bid > self.max_bid: # need to see if the price actually increased this round to be reimbursed
            reimbursements[round_max_bid_index] = (round_max_bid - self.max_bid) * self.reimbursement_rates[self.current_round]
            self.max_bid = round_max_bid
            self.max_bid_index = round_max_bid_index
            self.agent_max_bid_holder = self.max_bid_index == 0

        last_round = self.current_round + 1 == self.num_rounds # because we increment at the end anyway
        won = last_round and (self.max_bid >= self.reserve_price)
        
        # do we need all rewards or just agent?
        rewards = [0] * (self.num_opponents + 1)
        for bidder in range(self.num_opponents + 1):
            rewards [bidder] = self.reward(all_bids[bidder], reimbursements[bidder], last_round, won, all_vals[bidder])

        std_round = self.current_round / self.num_rounds # standardized round number for observation
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), std_round], dtype = np.float32)

        self.current_round += 1
        extra_info = { 
            "Other Agent Rewards" : rewards[1:] # for debugging?
        }
        
        return observation, rewards[0], last_round, False, extra_info # false is truncated

    def reward(self, bid, reimbursement, last_round, won, valuation): # is reward = utility?
        
        # based on round winning, abstention, and valuation (val is constant)

        if reimbursement == 0:
            if bid is None:
                return 0
            else: 
                return -(self.bid_cost)
            
        else:
            if last_round and won: 
                return valuation - bid + reimbursement - self.bid_cost
            else:
                return reimbursement - self.bid_cost
            
    # NEEDS TO BE UPDATED BIG TIME DOWNT THE ROAD!!!
    def opp_bids(self): 

        opp_bids = []
        for i in range(self.num_opponents):

            bid = self.valuation(self.opp_signals, i, self.max_bid) # arbitrary opponent bidding strategy, based on the max bid
            leader = self.max_bid_index == i + 1 # max bid index already includes the agent at 0

            if not leader: # step already ensures that the bid is a valid size
                opp_bids.append(bid)
            else: 
                opp_bids.append(None)

        self.opponents = opp_bids
        return opp_bids

    # this function may be updated to be more complex!!
    # currently used for both agent and opponents
    def valuation(self, signals, bidder_num, stat): 
        return np.clip(signals[bidder_num] + (stat * self.valuation_weight), 0.0, 1.0) # no randomness, valuation func is deterministic: if not, what's the point of a signal?

    # delete or overwrite this?
    def render(self):
        if self.render_mode == "human":
            print(f"Round Number: {int(self.current_round)}")
            print(f"Agent Signal: {self.agent_signal}")
            print(f"Opponent Signals: {self.opp_signals}")
            print(f"Highest Bid: {self.max_bid}")
        


    








        

        


