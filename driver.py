import numpy as np
from environment import Apra_env
from distinguished_agent import Distinguished_agent
import plots

# HYPERPARAMETERS

# environment
NUM_ROUNDS = 5
NUM_OPPONENTS = 3
RESERVE_PRICE = 0.2
BID_COST = 0.08
# reimbursement rate is begin swept and thus instantiated per run eleswhere
# REIMBURSEMENT_RATES = [0.25] * NUM_ROUNDS # one per round, most important variable in this research


SIGNAL_NOISE = 0.1

# agent
LEARNING_RATE = 0.0005
DISCOUNT_RATE = 0.99 # keeping it high so the last round (where win happens) is weighted heavily early on
REPLAY_BUFF_SIZE = 10000
BATCH_PULL_SIZE = 128
NUM_AVAILABLE_BIDS = 101 # ex. 101 => 0.00, 0.01, ... 1.0; we need the extra 1 for 1.00
OBS_SIZE = 4
EPS_DECAY = 0.9985

# training
NUM_EPISODES = 30000
TARGET_UPDATE_FREQ = 10
WARMUP_FRACTION = 2/3
WARMUP_EPISODES = int(NUM_EPISODES * WARMUP_FRACTION)
# where we can make a curriculum stages dictionary to iterate over
    # can include different number of opponents, rounds, etc. in each stage

# reimbursement sweep 
hypes_per_reimburse = {}
bids_per_reimburse = {}
vals_per_reimburse = {}
max_bids_per_reimburse = {}
revenues_per_reimburse = {}

REIMBURSEMENT_SWEEPS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

for r in REIMBURSEMENT_SWEEPS:

    REIMBURSEMENT_RATES = [r] * NUM_ROUNDS

    win_log = []
    reward_log = []
    revenue_log = []
    hype_log = [] # hype per episode--hype per round is in google sheet
    max_bids = []

    agent = Distinguished_agent(LEARNING_RATE, DISCOUNT_RATE, REPLAY_BUFF_SIZE, BATCH_PULL_SIZE, NUM_AVAILABLE_BIDS, OBS_SIZE)
    env = Apra_env(NUM_ROUNDS, NUM_OPPONENTS, RESERVE_PRICE, REIMBURSEMENT_RATES, BID_COST, SIGNAL_NOISE)

    with open("training_log.txt", "w") as f:

        # logs for plotting
        losses = []
        bids = []
        vals = []

        agg_round = 0

        for episode in range(1, NUM_EPISODES + 1):

            observation, info = env.reset() # info is optional for debugging

            total_reward = 0
            won = False

            # auctioneer variables--what we are testing for
            hype = 0.0

            for round_num in range(NUM_ROUNDS):

                action_raw_index = agent.choose_action(observation)
                action_discrete = action_raw_index / (NUM_AVAILABLE_BIDS - 1) # index is the nueron number. we need to get it to a float [0,1); -1 is so that it isn't 1. ex. index 37 / 40 bid options would be high percentile bid

                next_observation, reward, terminated, _ = env.step(np.array([action_discrete]))
                agent.store_transition(observation, action_raw_index, reward, next_observation, terminated)

                loss = agent.update_policy() # able to be called with unfull buffer, because the agent class handles it
                observation = next_observation
                total_reward += reward

                hype += env.max_bid

                agg_round += 1
                if agg_round >= BATCH_PULL_SIZE: # only want to check loss once it begins to get calculated
                    losses.append(loss.item())

                if agg_round >= WARMUP_EPISODES * NUM_ROUNDS: 
                    if round_num == NUM_ROUNDS - 1:
                        max_bids.append(observation[1])
                    bids.append(action_discrete)
                    vals.append(env.age_val)

                if terminated:
                    won = env.agent_max_bid_holder and env.max_bid >= RESERVE_PRICE # only check win/loss at end of auction

                if episode >= WARMUP_EPISODES and episode % 100 == 0:
                    opp_str = [f"{b:.3f}" if b is not None else "None" for b in env.opp_bid_vals]
                    f.write(f"episode: {episode} | round: {round_num + 1} | strat: {agent.strat} | valution: {env.age_val:.3f} | bid: {action_discrete:.3f} | opps: {opp_str} | won: {won} | reward: {total_reward:.4f} | epsilon: {agent.epsilon:.4f} | hype: {hype:.3f}\n")

            revenue = env.max_bid - env.total_reimbursement # max bid at time of last round is the winning bid

            if episode >= WARMUP_EPISODES and episode % 100 == 0:
                f.write(f"\n")

            if episode >= WARMUP_EPISODES: # we only want to log when exploiting
                hype_log.append(hype)
                revenue_log.append(revenue)
                win_log.append(won)
                reward_log.append(total_reward)

            if episode % TARGET_UPDATE_FREQ == 0:
                agent.decay_eps(decay = EPS_DECAY) # only decays after warmup (agent class handles this)
                agent.update_target()

        # 1500 and 100 arbitrarily chosen
        plots.windowed_metric(losses, 1500, "Average Loss")
        plots.windowed_metric(reward_log, 100, "Average Reward")
        plots.windowed_metric(win_log, 100, "Win Rate")
        plots.windowed_metric(revenue_log, 100, "Average Revenue")
        plots.bid_ratio_round(bids, vals, NUM_ROUNDS, 100)

        print(f"Avg. Win Rate: {np.mean(win_log):.4f}")
        print(f"Avg. Reward: {np.mean(reward_log):.4f}")
        print(f"Avg. Revenue: {np.mean(revenue_log):.4f}")
        print(f"Avg. Hype: {np.mean(hype_log):.4f}")

    hypes_per_reimburse[r] = hype_log
    bids_per_reimburse[r] = bids
    vals_per_reimburse[r] = vals
    max_bids_per_reimburse[r] = max_bids
    revenues_per_reimburse[r] = revenue_log

print("hi")
plots.one_metric_sweep(hypes_per_reimburse, "Reimbursement Rate", "Average Hype")
plots.bid_val_round_per_reimburse(bids_per_reimburse, vals_per_reimburse, NUM_ROUNDS)



