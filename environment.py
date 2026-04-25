import numpy as np
# parent class
import gymnasium as gym
# used for action/observation data checking
from gymnasium import spaces

# env class, inherits from gym
class Apra_env(gym.Env):
    
    # constructor method, inheriting environment
    def __init__(self, num_rounds, num_opponents, reserve_price, reimbursement_rates, bid_cost, render_mode = None):
        super().__init__()

        # auction-env specific variables, as well as the agent signals and reward tracker
        self.num_rounds = num_rounds
        self.current_round = 0
        self.num_opponents = num_opponents
        self.reserve_price = reserve_price
        self.reimbursement_rates = reimbursement_rates
        self.bid_cost = bid_cost
        self.highest_bid = 0.0
        self.agent_signal = 0.0
        self.opp_signals = np.zeros(self.num_opponents)
        self.agent_tot_reward = 0.0
        self.render_mode = render_mode

        # define action and observation spaces
            # all already 0, 1 except round # which we normalize
            # Box: all valid cartesian products
            # action: bid // observation: signal, highest bid, round #
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32)
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (3,), dtype = np.float32)


    # reset method to start from scratch
    def reset(self, seed = None, options = None):

        # reset to both to 0, no need to scale round if 0
        self.current_round = 0r
        self.highest_bid = 0.0

        # seed for rng
        super().reset(seed = seed)

        # check if special fixed signal for agent (for debugging)
        if options and options.get("fixed_signal") is not None:
            self.agent_signal = options["fixed_signal"]
        else:
            self.agent_signal = self.np_random.uniform(0.0, 1.0)

        # randomized opponent signals
        self.opp_signals = self.np_random.uniform(0.0, 1.0, size = self.num_opponents)

        # make observation space to give to the agent
        observation = np.array([self.agent_signal, self.highest_bid, self.current_round], dtype = np.float32)

        # extra info for debugging
        info = {
            "agent_signal": self.agent_signal,
            "opp_signals": self.opp_signals
        }

        return observation, info
    
    # render basic text to the terminal for tracking when we choose to
    def render(self):
        if self.render_mode == "human":
            print(f"Round Number: {int(self.current_round * self.num_rounds)}")
            print(f"Agent Signal: {self.agent_signal}")
            print(f"Opponent Signals: {self.opp_signals}")
            print(f"Highest Bid: {self.highest_bid}")
            print(f"Agent Reward: {self.agent_tot_reward}")








        

        


