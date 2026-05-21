import numpy as np
from environment import Apra_env
from distinguished_agent import Distinguished_agent
    
# HYPERPARAMETERS (can obviously all be tweaked)

# environment
NUM_ROUNDS = 5
NUM_OPPONENTS = 3  
RESERVE_PRICE = 0.2
REIMBURSEMENT_RATES = [0.25] * NUM_ROUNDS # need one per round
BID_COST = 0.05
SIGNAL_NOISE = 0.1
VALUATION_WEIGHT = 0.25
LOSS_ADDITION = 0.05
BID_THRESH = 0.1

# agent
LEARNING_RATE = 0.001
DISCOUNT_RATE = 0.95
REPLAY_BUFF_SIZE = 10000
BATCH_PULL_SIZE = 64
NUM_AVAILABLE_BIDS = 101 # ex. 101 => 0.00, 0.01, ... 1.0; we need the extra 1 for 1.00
OBS_SIZE = 4 # for agent and environment

# training
NUM_EPISODES = 20000
TARGET_UPDATE_FREQ = 10
# where we can make a curriculum stages dictionary to iterate over
    # can include different number of opponents, rounds, etc. in each stage

# BOOKKEEPING AND INSTANTIATION

# general bookkeeping--can always add more
win_log = [] # binary
reward_log = []

agent = Distinguished_agent(LEARNING_RATE, DISCOUNT_RATE, REPLAY_BUFF_SIZE, BATCH_PULL_SIZE, NUM_AVAILABLE_BIDS, OBS_SIZE)
env = Apra_env(NUM_ROUNDS, NUM_OPPONENTS, RESERVE_PRICE, REIMBURSEMENT_RATES, BID_COST, SIGNAL_NOISE, VALUATION_WEIGHT, OBS_SIZE, LOSS_ADDITION, BID_THRESH)

# LOOP PORTION

with open("training_log.txt", "w") as f:

    for episode in range(1, NUM_EPISODES + 1):
        observation, info = env.reset() # info is optional for debugging
        
        total_reward = 0
        won = False

        for round in range(NUM_ROUNDS):
            
            action_raw_index = agent.choose_action(observation)
            action_discrete = action_raw_index / (NUM_AVAILABLE_BIDS - 1) # index is the nueron number. we need to get it to a float [0,1); -1 is so that it isn't 1. ex. index 37 / 40 bid options would be high percentile bid

            next_observation, reward, terminated, truncated = env.step(np.array([action_discrete]), round)
            agent.store_transition(observation, action_raw_index, reward, next_observation, terminated)

            agent.update_policy() # able to be called with unfull buffer, because the agent class handles it
            observation = next_observation
            total_reward += reward

            if terminated or truncated:
                won = env.agent_max_bid_holder and env.max_bid >= RESERVE_PRICE # only check win/loss at end of auction

            # print(f"episode: {episode} | round: {round} | won: {won} | reward: {total_reward:.4f} | agent_signal: {action_discrete:.3f} | opp_bids: {env.opponents}")
            if episode % 100 == 0:
                opp_str = [f"{b:.3f}" if b is not None else "None" for b in env.opponents]
                f.write(f"episode: {episode} | round: {round + 1} | strat: {agent.strat} | valution: {env.age_val:.3f} | bid: {action_discrete:.3f} | opps: {opp_str} | won: {won} | reward: {total_reward:.4f} | epsilon: {agent.epsilon:.4f}\n")

        win_log.append(won)
        reward_log.append(total_reward)
        if episode % TARGET_UPDATE_FREQ == 0:
            agent.decay_eps() # only decays after warmup (agent class handles this)
            agent.update_target()

        
        












