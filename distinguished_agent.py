# imports for neural network, replay buffer, and math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

# neural network class
class QNetwork(nn.Module):

    # make network (sequential is blueprint)
    def __init__(self, input_size, output_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )

    # automatically called to run through the network and get the outputs
    def forward(self, inputs):
        return self.network(inputs)

# agent (brain) class
class distinguished_agent:

    # constructor taking in hyperparameters and setting up the replay buffer
    # can add more parameters to tweak as it gets more complex
    def __init__(self, learning_rate, discount_future_rate, replay_buff_size, batch_pull_size, num_bid_options):
        self.learning_rate = learning_rate
        self.discount_future_rate = discount_future_rate
        self.replay_buff_size = replay_buff_size
        self.batch_pull_size = batch_pull_size
        self.action_size = num_bid_options
        # should replace a lot of Miriam's bookkeeping
        self.replay_buffer = deque(maxlen=replay_buff_size)

        # set up the target and policy network--4 is the observation size
        self.policy_net = QNetwork(4, self.action_size)
        self.target_net = QNetwork(4, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # we only have the policy network learn, as we manually update the more-static target
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    # choose highest Q value bid
    def choose_action(self, observation):
        pass

    # add transition to replay buffer
    def store_transition(self, state, action, reward, next_state, done):
        pass

    # update the policy after every step
    def update_policy(self):
        pass

    # update the target after X episodes
    def update_target(self):
        pass