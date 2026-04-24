import numpy as np
# parent class
import gymnasium as gym
# used for action/observation data checking
from gymnasium import spaces

# env class, inherits from gym
class Apra_env(gym.Env):
    
    # constructor method, inheriting environment
    def __init__(self, num_rounds, num_opponents, reserve_pice, reimbursement_rates, bid_cost):
        super.__init__()

        # auction-env specific variales
        self.num_rounds = num_rounds
        self.num_opponents = num_opponents
        self.reserve_pice = reserve_pice
        self.reimbursement_rates = reimbursement_rates
        self.bid_cost = bid_cost

        # define action and observation spaces
            # all already 0, 1 except round # which we normalize
            # Box: all valid cartesian products
            # action: bid // observation: highest bid, signal, round #
        self.action_space = spaces.Box(low = 0.0, high = 1.0, shape = (1,), dtype = np.float32)
        self.observation_space = spaces.Box(low = 0.0, high = 1.0, shape = (3,), dtype = np.float32)


