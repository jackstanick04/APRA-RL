import numpy as np
import gymnasium as gym
from gymnasium import spaces

class Apra_env(gym.Env):
    
    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, signal_noise, valuation_weight, obs_size, loss_addition, bid_thresh, render_mode = None):
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
        self.loss_addition = loss_addition
        self.render_mode = render_mode
        self.bid_thresh = bid_thresh

        # auction states (reset per episode)
        self.current_round = 0
        self.max_bid = 0.0
        self.max_bid_index = -1
        self.agent_max_bid_holder = False
        self.agent_signal = 0.0
        self.opp_signals = np.zeros(self.num_opponents)
        self.opp_loss_streaks = np.zeros(self.num_opponents)
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

    def step(self, action, round_num):
        
        agent_bid = None if (action[0] < 0.01  or action[0] < self.max_bid ) else action[0] # punishes him for bidding invalid amount, because it triggers the wrong abstention penalty in reward function

        opp_bids = self.opp_bids()
        all_bids = [agent_bid] + opp_bids
        round_max_bid = max((bid for bid in all_bids if bid is not None), default = 0.0)
        round_max_bid_index = all_bids.index(round_max_bid) if round_max_bid > 0.0 else -1 # if everyone asbtains, nobody gets reimbursed

        was_leader = self.agent_max_bid_holder
        reimbursements = [0] * (self.num_opponents + 1)
        if round_max_bid > self.max_bid: # need to see if the price actually increased this round to be reimbursed
            reimbursements[round_max_bid_index] = (round_max_bid - self.max_bid) * self.reimbursement_rates[self.current_round]
            self.max_bid = round_max_bid
            self.max_bid_index = round_max_bid_index
            self.agent_max_bid_holder = self.max_bid_index == 0

        # agent is index 0
        all_signals = [self.agent_signal] + list(self.opp_signals)
        all_vals = [self.valuation(all_signals, i, self.max_bid) for i in range(self.num_opponents + 1)]
        self.age_val = all_vals[0]

        last_round = self.current_round + 1 == self.num_rounds # because we increment at the end anyway
        won = last_round and (self.max_bid >= self.reserve_price)
        
        # only agent's reward, because the opponents aren't learning
        reward = self.reward(round_num, last_round, won, self.max_bid, agent_bid, all_vals[0], reimbursements[0], self.agent_max_bid_holder, was_leader, self.bid_thresh)

        std_round = self.current_round / self.num_rounds # standardized round number for observation
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), std_round], dtype = np.float32)

        self.current_round += 1
        
        return observation, reward, last_round, False # false is truncated
    
    # this function may be updated to be more complex!!
    # currently used for both agent and opponents
    def valuation(self, signals, bidder_num, stat): 
        return np.clip(signals[bidder_num] + (stat * self.valuation_weight), 0.0, 1.0) # no randomness, valuation func is deterministic: if not, what's the point of a signal?
    
    # NEEDS TO BE UPDATED BIG TIME DOWNT THE ROAD!!!
    # SHOULD BE SHADING, BUT DOING IT WITHOUT TO BE EVEN STRICTER ON MONKEE
    def opp_bids(self): 

        opp_bids = []
        for i in range(self.num_opponents):

            leader = self.max_bid_index == i + 1 # max bid index already includes the agent at 0

            value = self.valuation(self.opp_signals, i, self.max_bid)
            loss_boost = self.loss_addition * self.opp_loss_streaks[i] # if max bid is the same and they have been losing, we need to bid more
            bid = np.clip(value + loss_boost, 0.0, 1.0)

            if not leader: # step already ensures that the bid is a valid size
                if bid <= self.max_bid:
                    opp_bids.append(None) # don't increment loss because we've given up. we aren't losing, we're just done 
                else:
                    opp_bids.append(bid)
                    self.opp_loss_streaks[i] += 1 # assume we lose, if we win the next iteration fixes this

            else: 
                opp_bids.append(None)
                self.opp_loss_streaks[i] = 0

        self.opponents = opp_bids
        return opp_bids

    # reward function depends on subfunctions per round number
    def reward(self, round_num, last_round, won, winning_bid, agent_bid, valuation, reimbursement, agent_leader, was_leader, bid_thresh):
        if round_num == 0:
            return self.first_round_rew(agent_bid, valuation, reimbursement)
        elif last_round:
            return self.last_round_rew(won, was_leader, winning_bid, agent_bid, valuation, reimbursement, bid_thresh)
        else:
            return self.other_round_rew(agent_leader, was_leader, winning_bid, agent_bid, valuation, reimbursement, bid_thresh)

    def first_round_rew(self, agent_bid, valuation, reimbursement):
        if agent_bid is None:
            return -.2 # agent must be penalized for abstaining in round 1, as it is never gto
        
        return reimbursement - self.bid_cost - max(agent_bid - valuation, 0) # punishes for bidding > valuation

    # there's probably a cleaner way to make this if skeleton
    def last_round_rew(self, won, was_leader, winning_bid, agent_bid, valuation, reimbursement, bid_thresh):
        if was_leader:
            if agent_bid is not None:
                return -.2 # agent needs to be penalized for not abiding gto abstention
            elif won:
                return valuation - winning_bid # no reimbursement or bid cost because that happened last round
            else:
                return 0
            
        else:
            if won: 
                return valuation - winning_bid + reimbursement - self.bid_cost
            else: # if he wasn't leading and lost, he is rewarded based on his action and if he had a chance
                if np.abs(valuation - winning_bid) > bid_thresh or winning_bid == 1.0: # check if auction already won
                    if agent_bid is None:
                        return 0
                    else:
                        return -self.bid_cost - max(agent_bid - valuation, 0)
                else:
                    if agent_bid is None:
                        return -.2 # agent must be punished for abstaining when he had a chance
                    else:
                        return -self.bid_cost - max(agent_bid - valuation, 0)

    # very similar to the last round function
    def other_round_rew(self, agent_leader, was_leader, winning_bid, agent_bid, valuation, reimbursement, bid_thresh):
        if was_leader:
            if agent_bid is not None:
                return -.2 # agent needs to be penalized for not abiding gto abstention
            else:
                return 0
            
        else:
            if agent_leader:
                return reimbursement - self.bid_cost - max(agent_bid - valuation, 0)
            else: # if he wasn't leading and lost, he is rewarded based on his action and if he had a chance
                if np.abs(valuation - winning_bid) > bid_thresh or winning_bid == 1.0: # if auction is won, he should abstain
                    if agent_bid is None:
                        return 0
                    else:
                        return -self.bid_cost - max(agent_bid - valuation, 0)
                else:
                    if agent_bid is None:
                        return -.5 # agent must be punished for abstaining when he had a chance
                    else:
                        return -self.bid_cost - max(agent_bid - valuation, 0)

    # delete or overwrite this?
    def render(self):
        if self.render_mode == "human":
            print(f"Round Number: {int(self.current_round)}")
            print(f"Agent Signal: {self.agent_signal}")
            print(f"Opponent Signals: {self.opp_signals}")
            print(f"Highest Bid: {self.max_bid}")

    # OLD REWARD FUNCTION CODE
     # if last_round:
        #     if won:
        #         return valuation - winning_bid + reimbursement - (self.bid_cost if bid is not None else 0)
        #     else:
        #         if bid is None:
        #             return 0
        #         else:
        #             return -(self.bid_cost)
        # else:
        #     if bid is None:
        #         return 0
        #     else:
        #         return reimbursement - self.bid_cost
        


    








        

        


