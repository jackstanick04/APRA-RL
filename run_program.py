import dataclasses
from configuration import Configuration
from driver import run_trial
from sweeper import one_d_sweep, two_d_sweep
import plots
import matplotlib.pyplot as plt

# SAMPLE PLOTTING EXAMPLES JUST TO VERIFY IT WORKS

type = "one_d_sweep"

base_config = Configuration()

if type == "non_sweep":
    config = dataclasses.replace(base_config, bid_cost = 0.2, num_episodes = 1000)
    logs = run_trial(config)
    fig = plots.windowed_metric(logs, 10, "reward")
    plt.show()

elif type == "one_d_sweep":
    config = dataclasses.replace(base_config, num_episodes = 1000)
    results = one_d_sweep(config, "bid_cost", values = [0.0, 0.05, 0.1])
    fig = plots.windowed_metric_one_sweep(results, "bid_cost", 10, "reward")
    plt.show()

elif type == "two_d_sweep":
    config = dataclasses.replace(base_config, num_episodes = 1000)
    results = two_d_sweep(config, "bid_cost", "reserve_price", values_one = [0.0, 0.05, 0.1], values_two = [0.1, 0.2, 0.3])
    fig = plots.windowed_metric_two_sweep(results, "bid_cost", "reserve_price", 10, "reward")
    plt.show()


