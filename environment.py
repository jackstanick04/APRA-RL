import numpy as np
import gymnasium as gym


class Apra_env(gym.Env):

    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, signal_noise):
        super().__init__()

        # auction parameters (fixed)
        self.num_rounds = num_rounds
        self.num_opponents = num_opponents
        self.reserve_price = reserve_price
        self.reimbursement_rates = reimbursement_rates
        self.bid_cost = bid_cost
        self.signal_noise = signal_noise
        self.opp_bid_vals = []

        # auction states (reset per episode)
        self.current_round = 0
        self.max_bid = 0.0
        self.agent_max_bid_holder = False
        self.agent_signal = 0.0
        self.opp_signals = np.zeros(self.num_opponents)
        self.total_reimbursement = 0.0
        self.age_val = 0.0

    def reset(self, seed = None, options = None):

        # only auction state variables reset
        self.current_round = 0
        self.max_bid = 0.0
        self.agent_max_bid_holder = False
        self.total_reimbursement = 0.0
        self.opp_bid_vals = []
        super().reset(seed = seed) # seed for agent signal debugging

        # make a base value, then add some noise to each person (keep between 0 and 1)
        true_value = self.np_random.uniform(0.0, 1.0)
        self.opp_signals = np.clip(true_value + self.np_random.standard_normal(self.num_opponents) * self.signal_noise, 0.0, 1.0)
        if options and options.get("fixed_signal") is not None:
            self.agent_signal = options["fixed_signal"]
        else:
            self.agent_signal = np.clip(true_value + self.np_random.standard_normal() * self.signal_noise, 0.0, 1.0)

        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), 0.0], dtype = np.float32)

        info = {
            "agent_signal": self.agent_signal,
            "opp_signals": self.opp_signals
        }

        return observation, info

    def step(self, action):

        agent_bid = None if action[0] < 0.01 else action[0]
        self.age_val = self.valuation()

        self.opp_bid_vals = self.opp_bids()
        all_bids = [agent_bid] + self.opp_bid_vals
        round_max_bid = max((bid for bid in all_bids if bid is not None), default = 0.0)
        round_max_bid_index_list = [i for i, bid in enumerate(all_bids) if bid == round_max_bid]

        reimburse = 0
        age_reimburse = 0
        if round_max_bid > self.max_bid: # need to see if the price actually increased this round to be reimbursed

            reimburse = (round_max_bid - self.max_bid) * self.reimbursement_rates[self.current_round]
            winning_index = np.random.choice(round_max_bid_index_list)
            if winning_index == 0: # agent is only reimbursed if they win the random tie breaker
                age_reimburse = reimburse
            self.total_reimbursement += reimburse

            self.max_bid = round_max_bid
            self.agent_max_bid_holder = winning_index == 0 # only max_bid_holder if randomly chosen out of the max bidders

        last_round = self.current_round + 1 == self.num_rounds # because we increment at the end anyway
        won = last_round and (self.max_bid >= self.reserve_price)
        agent_won = won and self.agent_max_bid_holder

        # only agent's reward, because the opponents aren't learning
        reward = self.reward(agent_won, agent_bid, self.max_bid, self.age_val, age_reimburse)

        std_round = self.current_round / self.num_rounds # standardized round number for observation
        observation = np.array([self.agent_signal, self.max_bid, float(self.agent_max_bid_holder), std_round], dtype = np.float32)

        self.current_round += 1

        return observation, reward, last_round, False # false is truncated

    def valuation(self):
        return self.agent_signal

    def reward(self, agent_won, agent_bid, winning_bid, agent_valuation, reimbursement):
        win_bonus = agent_valuation - winning_bid if agent_won else 0 # if the agent wins, winning_bid = agent_bid

        if agent_bid is not None:
            return win_bonus + reimbursement - self.bid_cost
        return win_bonus # abstention gives 0 unless it's the last round and agent won

    def opp_bids(self): # placeholder signature for now
        return [0] * self.num_opponents
