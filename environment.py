import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Apra_env(gym.Env):
    
    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, signal_noise, valuation_weight, obs_size, loss_addition):
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
        self.loss_addition = loss_addition # for opponents

        # auction states (reset per episode)
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False
        self.agent_signal = 0.0
        self.opp_signals = np.zeros(self.num_opponents)
        self.opp_loss_streaks = np.zeros(self.num_opponents)
        self.total_reimbursement = 0.0
        self.opponents = [] # for debugging
        self.age_val = 0.0 # for debugging

        # Box spaces ensure that all observation and action values are in [0,1] range
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32) # action: bid
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (self.obs_size,), dtype = np.float32) # observation: agent_signal, max_bid, agent_max_bid_holder, round_num (gets scaled to [0,1])

    def reset(self, seed = None, options = None):

        # only auction state variables reset
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False
        self.opp_loss_streaks = np.zeros(self.num_opponents)
        self.total_reimbursement = 0.0
        super().reset(seed = seed) # seed for agent signal debugging

        # make a base value, then add some noise to each person (keep between 0 and 1)
        true_value = self.np_random.uniform(0.0, 1.0)
        self.opp_signals = np.clip(true_value + self.np_random.standard_normal(self.num_opponents) * self.signal_noise, 0.0, 1.0)
        # check if special fixed signal for agent (for debugging)
        if options and options.get("fixed_signal") is not None:
            self.agent_signal = options["fixed_signal"]
        else: 
            self.agent_signal = np.clip(true_value + self.np_random.standard_normal() * self.signal_noise, 0.0, 1.0)

        observation = np.array([self.agent_signal, self.max_bid,float(self.agent_max_bid_holder), 0.0], dtype = np.float32)

        # extra info for debugging
        info = {
            "agent_signal": self.agent_signal,
            "opp_signals": self.opp_signals
        }

        return observation, info

    def step(self, action, round_num):
        
        agent_bid = None if action[0] < 0.01 else action[0] # punishes him for bidding invalid amount, because it triggers the wrong abstention penalty in reward function
        self.age_val = self.valuation(self.agent_signal, self.max_bid, round_num)

        opp_bids = self.opp_bids(round_num) # i think we can use the valuation calculations from earlier to shorten the opp_bids method
        all_bids = [agent_bid] + opp_bids
        round_max_bid = max((bid for bid in all_bids if bid is not None), default = 0.0) # probably could be made more efficient
        round_max_bid_index_list = [i for i, bid in enumerate(all_bids) if bid == round_max_bid and round_max_bid is not None] 

        # not sure if we need to calculate the reimbursements for the opps yet
        # AGENT IS ALWAYS INDEX 0
        reimbursements = [0] * (self.num_opponents + 1)
        if round_max_bid > self.max_bid: # need to see if the price actually increased this round to be reimbursed

            reimburse = (round_max_bid - self.max_bid) * self.reimbursement_rates[self.current_round]
            winning_index = np.random.choice(round_max_bid_index_list) # randomize the winner if multiple tie the highest bid
            reimbursements[winning_index] = reimburse
            self.total_reimbursement += reimburse

            self.max_bid = round_max_bid
            self.max_bid_index = winning_index
            self.agent_max_bid_holder = winning_index == 0 # only max_bid_holder if randomly chosen out of the max bidders

        last_round = self.current_round + 1 == self.num_rounds # because we increment at the end anyway
        won = last_round and (self.max_bid >= self.reserve_price)
        agent_won = won and self.agent_max_bid_holder
        
        # only agent's reward, because the opponents aren't learning
        reward = self.reward(agent_won, agent_bid, self.max_bid, self.age_val, reimbursements[0])

        std_round = self.current_round / self.num_rounds # standardized round number for observation
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), std_round], dtype = np.float32)

        self.current_round += 1
        
        return observation, reward, last_round, False # false is truncated
    
    # both agent and opponents
    # can be updated down the line for better interdependence and theory
    def valuation(self, signal, stat, round_num): 

        if round_num == 0:
            return signal # don't want first round valuation to be cut (current max bid is 0)

        return np.clip(((1 - self.valuation_weight) * signal) + (stat * self.valuation_weight), 0.0, 1.0) # no randomness, but deterministic: if not, what's the point of a signal?
    
    # NEEDS TO BE UPDATED BIG TIME DOWNT THE ROAD!!!
    # SHOULD BE SHADING, BUT DOING IT WITHOUT TO BE EVEN STRICTER ON MONKEE
    def opp_bids(self, round_num): 

        opp_bids = []
        for i in range(self.num_opponents):

            pass
            
            
            
            # OLD OPP_BIDS FUNCTION
            # leader = self.max_bid_index == i + 1 # max bid index already includes the agent at 0

            # value = self.valuation(self.opp_signals[i], self.max_bid, round_num)
            # loss_boost = self.loss_addition * self.opp_loss_streaks[i] # if max bid is the same and they have been losing, we need to bid more
            # bid = np.clip(value + loss_boost, 0.0, 1.0)

            # if not leader: # step already ensures that the bid is a valid size
            #     if bid <= self.max_bid:
            #         opp_bids.append(None) # don't increment loss because we've given up. we aren't losing, we're just done 
            #     else:
            #         opp_bids.append(bid)
            #         self.opp_loss_streaks[i] += 1 # assume we lose, if we win the next iteration fixes this

            # else: 
            #     opp_bids.append(None)
            #     self.opp_loss_streaks[i] = 0

        self.opponents = opp_bids
        return opp_bids

    def reward(self, agent_won, agent_bid, winning_bid, agent_valuation, reimbursement):
        
        win_bonus = agent_valuation - winning_bid if agent_won else 0 # if the agent wins, winning_bid = agent_bid

        if agent_bid is not None:
            return win_bonus + reimbursement - self.bid_cost
        return win_bonus # abstention gives 0 unless it's the last round and agent won



    








        

        


