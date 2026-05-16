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

        # linear connections between layers of appropriate sizes with ReLU as activation; hardcoding the architecture for now
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
class Distinguished_agent:

    # constructor taking in hyperparameters and setting up the replay buffer
    # can add more parameters to tweak as it gets more complex
    def __init__(self, learning_rate, discount_future_rate, replay_buff_size, batch_pull_size, num_bid_options, obs_size):
        self.learning_rate = learning_rate
        self.discount_future_rate = discount_future_rate
        self.replay_buff_size = replay_buff_size
        self.batch_pull_size = batch_pull_size
        self.action_size = num_bid_options
        self.epsilon = 1.0 # want to always explore at first
        self.obs_size = obs_size
        # should replace a lot of Miriam's bookkeeping
        self.replay_buffer = deque(maxlen=replay_buff_size)

        # set up the target and policy network
        self.policy_net = QNetwork(self.obs_size, self.action_size)
        self.target_net = QNetwork(self.obs_size, self.action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        # we only have the policy network learn, as we manually update the more-static target
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    # choose highest Q value bid; returns index of the discretized bid, not the acutal bid value
    def choose_action(self, observation):
        
        # exploitation v. exploration based on epsilon
        choice = np.random.uniform(0, 1)
        # explore
        if choice <= self.epsilon:
            return np.random.randint(0, self.action_size)

        # exploit
        obser_tens = torch.FloatTensor(observation) # neural networks need tensors, not arrays
        with torch.no_grad(): # do not need to track the gradients when only calculating outputs 
            q_values = self.policy_net(obser_tens)
        return q_values.argmax().item() # returns the maximum q value index as an integer

    # add transition to replay buffer; deque structure automatically kicks the other elements off
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    # update the policy after every step
    def update_policy(self):
        
        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        # select a random batch of transitions, storing all the states, etc. together
        batches = random.sample(self.replay_buffer, self.batch_pull_size)
        states, actions, rewards, next_states, dones = zip(*batches) # * splits up all the batches into tuples; zip then pairs each tuple's values at an index together (makes sense)

        # convert to tensors so we can run calculations
        states_tens = torch.FloatTensor(np.array(states))
        actions_tens = torch.LongTensor(np.array(actions, dtype = np.int64))
        rewards_tens = torch.FloatTensor(np.array(rewards))
        next_states_tens = torch.FloatTensor(np.array(next_states))
        dones_tens = torch.FloatTensor(np.array(dones))

        # calculate the q values for each transition, and store the gradients
        pred_qs = self.policy_net(states_tens).gather(1, actions_tens.unsqueeze(1)).squeeze(1) # not really sure what squeeze is doing?

        # calculate the target q values for each transition -- don't need to store gradients, because not updating the target here
        with torch.no_grad():
            future_target_qs = self.target_net(next_states_tens).max(1)[0] # 1 is all the actions, 0 is just the value not index

        # calculate the bellman value; if done, future is 0
        target_qs = rewards_tens + self.discount_future_rate * future_target_qs * (1 - dones_tens)

        # calculate loss and backpropagate the policy network
        loss = self.loss_function(pred_qs, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # the actual updating

    # update the target after X episodes; tau is the weight of change
    def update_target(self, tau = 0.05):
        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        # zip pairs each weight for each network together
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            # then use the tau weighted sum to update the target slowly
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)

    def decay_eps(self, min_eps = 0.01, decay = 0.995):
        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        self.epsilon = max(min_eps, self.epsilon * decay)