import numpy as np
from environment import Apra_env
from distinguished_agent import Distinguished_agent

def run_trial(config):

    num_episodes = config.num_episodes
    target_update_freq = config.target_update_freq
    warmup_episodes = config.warmup_episodes

    env = Apra_env(
        num_rounds = config.num_rounds,
        num_opponents = config.num_opponents,
        reserve_price = config.reserve_price,
        reimbursement_rates = config.reimbursement_rates,
        bid_cost = config.bid_cost,
        signal_noise = config.signal_noise
    )

    agent = Distinguished_agent(
        learning_rate = config.learning_rate,
        discount_future_rate = config.discount_rate,
        replay_buff_size = config.replay_buff_size,
        batch_pull_size = config.batch_pull_size, 
        num_bid_options = config.num_available_bids,
        obs_size = config.observation_size
    )

    agg_round = 0
    logs = {
        # per round (handled in plotting)
        "losses": [],
        "bids": [],
        "vals": [],
        # per episode 
        "max_bids": [],
        "revenues": [],
        "hype": [],
        "wins": [],
        "reward": [],
    }

    for episode in range(1, num_episodes + 1):

        observation, info = env.reset() # info is optional for debugging

        total_reward = 0
        won = False
        hype = 0.0

        for round_num in range(config.num_rounds):

            action_raw_index = agent.choose_action(observation)
            action_discrete = action_raw_index / (config.num_available_bids - 1) # index is the nueron number. we need to get it to a float [0,1); -1 is so that it isn't 1. ex. index 37 / 40 bid options would be high percentile bid

            next_observation, reward, terminated, _ = env.step(np.array([action_discrete]))
            agent.store_transition(observation, action_raw_index, reward, next_observation, terminated)

            loss = agent.update_policy() # able to be called with unfull buffer, because the agent class handles it
            observation = next_observation
            total_reward += reward
            hype += env.max_bid

            agg_round += 1
            if agg_round >= config.batch_pull_size: # only want to check loss once it begins to get calculated
                logs["losses"].append(loss.item())

            if agg_round >= config.num_rounds * warmup_episodes: 
                if round_num == config.num_rounds - 1: 
                    logs["max_bids"].append(env.max_bid)
                logs["bids"].append(action_discrete)
                logs["vals"].append(env.age_val)

            if terminated:
                won = env.agent_max_bid_holder and env.max_bid >= config.reserve_price # only check win/loss at end of auction

        revenue = env.max_bid - env.total_reimbursement # max bid at time of last round is the winning bid

        if episode >= warmup_episodes: # we only want to log when exploiting
            logs["wins"].append(won)
            logs["reward"].append(total_reward)
            logs["hype"].append(hype)
            logs["revenues"].append(revenue)

        if episode % target_update_freq == 0:
            agent.decay_eps(decay = config.eps_decay) # only decays after warmup (agent class handles this)
            agent.update_target()

    return logs



