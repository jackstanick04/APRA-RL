import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):

    def __init__(self, input_size, output_size, hidden_layer_size, hidden_layer_amount): # makes parameterized NN skeleton
        super().__init__()

        layers = []
        prev_size = input_size

        for _ in range(hidden_layer_amount):
            layers.append(nn.Linear(prev_size, hidden_layer_size))
            layers.append(nn.ReLU())
            prev_size = hidden_layer_size
        layers.append(nn.Linear(prev_size, output_size))

        self.network = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.network(inputs)


class Distinguished_agent:

    def __init__(self, hidden_layer_size, hidden_layer_amount, learning_rate, discount_future_rate, replay_buff_size, batch_pull_size, num_bid_options, obs_size):
        self.discount_future_rate = discount_future_rate
        self.batch_pull_size = batch_pull_size
        self.action_size = num_bid_options
        self.epsilon = 1.0 # want to always explore at first
        self.replay_buffer = deque(maxlen=replay_buff_size)
        self.strat = "" # debugging exploration v. exploitation

        self.policy_net = QNetwork(obs_size, self.action_size, hidden_layer_size, hidden_layer_amount)
        self.target_net = QNetwork(obs_size, self.action_size, hidden_layer_size, hidden_layer_amount)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # target starts as copy

        # only have the policy network learn, as we manually update the more-static target
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_function = nn.MSELoss()

    def choose_action(self, observation):

        choice = np.random.uniform(0, 1)

        # explore
        if choice <= self.epsilon:
            self.strat = "explore"
            return np.random.randint(0, self.action_size)

        # exploit
        obser_tens = torch.FloatTensor(observation) # neural networks need tensors, not arrays
        with torch.no_grad(): # do not need to track the gradients when only calculating outputs
            q_values = self.policy_net(obser_tens)
        self.strat = "exploit"
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done)) # deque will automatically keep desired length

    def update_policy(self):

        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        batches = random.sample(self.replay_buffer, self.batch_pull_size)
        states, actions, rewards, next_states, dones = zip(*batches)

        # convert to tensors so we can run calculations
        states_tens = torch.FloatTensor(np.array(states))
        actions_tens = torch.LongTensor(np.array(actions, dtype = np.int64))
        rewards_tens = torch.FloatTensor(np.array(rewards))
        next_states_tens = torch.FloatTensor(np.array(next_states))
        dones_tens = torch.FloatTensor(np.array(dones))

        pred_qs = self.policy_net(states_tens).gather(1, actions_tens.unsqueeze(1)).squeeze(1)

        with torch.no_grad(): # don't need to store gradients, because not updating the target here
            future_target_qs = self.target_net(next_states_tens).max(1)[0]

        # bellman; if done, future is 0
        target_qs = rewards_tens + self.discount_future_rate * future_target_qs * (1 - dones_tens)

        # calculate loss and backpropagate the policy network
        loss = self.loss_function(pred_qs, target_qs)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def update_target(self, tau = 0.05):

        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data) # tau updates target slowly

    def decay_eps(self, min_eps = 0.01, decay = 0.995):
        # if the replay buffer doesn't have enough data, don't worry about training yet
        if len(self.replay_buffer) < self.batch_pull_size:
            return

        self.epsilon = max(min_eps, self.epsilon * decay)
