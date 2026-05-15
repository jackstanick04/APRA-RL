# import agent and environment
import numpy as np
from environment import Apra_env
from distinguished_agent import Distinguished_agent
    
# HYPERPARAMETERS

# environment
NUM_ROUNDS = -1
NUM_OPPONENTS = -1  
RESERVE_PRICE = -1
REIMBURSEMENT_RATES = # one per round
BID_COST = -1

# agent
LEARNING_RATE = -1
DISCOUNT_RATE = -1
REPLAY_BUFF_SIZE = -1
BATCH_PULL_SIZE = -1
NUM_AVAILABLE_BIDS = -1 # ex. 100 => 0.00, 0.01, ... 1.0

# main training
NUM_EPISODES = -1
TARGET_UPDATE_FREQ = -1
# where we can make a curriculum stages dictionary to iterate over
    # can include different number of opponents, rounds, etc.

# BOOKKEEPING AND INSTANTIATION

# general bookkeeping--we can always add more
win_log = [] # binary
reward_log = []

# instatiate the agent and environment
agent = Distinguished_agent(LEARNING_RATE, DISCOUNT_RATE, REPLAY_BUFF_SIZE, BATCH_PULL_SIZE, NUM_AVAILABLE_BIDS)
env = Apra_env(NUM_ROUNDS, NUM_OPPONENTS, RESERVE_PRICE, REIMBURSEMENT_RATES, BID_COST)

# LOOP PORTION

# loop every episode, reseting the environment each time
for episode in range(1, NUM_EPISODES + 1):
    observation, info = env.reset() # info is optional for debugging
    # total reward and win flag for bookkeeping
    total_reward = 0
    won = False

    # loop through each round of the auction (pretty logical)
    for round in range(NUM_ROUNDS):
        
        # get the discretized bid
        action_raw_index = agent.choose_action(observation)
        action_discrete = action_raw_index / (NUM_AVAILABLE_BIDS - 1) # index is the nueron number. we need to get it to a float [0,1); -1 is so that it isn't 1. ex. index 37 / 40 bid options would be high percentile bid

        # step the environment based on this bid and store transition to the buffer
        next_observation, reward, terminated, truncated, info = env.step(np.array([action_discrete]))
        agent.store_transition(observation, action_raw_index, reward, next_observation, terminated)

        # update policy and move to the next state
        agent.update_policy()
        observation = next_observation
        total_reward += reward

        # check if the game is done
        if terminated or truncated:
            won = env.agent_max_bid_holder and env.max_bid > RESERVE_PRICE
            break

    # update the logs and decay epislon every 10th episode
    win_log.append(won)
    reward_log.append(total_reward)
    if episode % TARGET_UPDATE_FREQ == 0:
        agent.decay_eps()
        agent.update_target()

    












