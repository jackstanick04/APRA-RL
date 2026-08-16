import dataclasses
from configuration import Configuration
from driver import run_trial
from sweeper import one_d_sweep, two_d_sweep
import plots # needs to be modernized to the current structure

type = "one_d_sweep" # where type of sweep/run is selected

base_config = Configuration()

if type == "non_sweep":
    # config = dataclasses.replace(base_config)
    # logs = run_trial(base_config)
    pass

elif type == "one_d_sweep": # just current example values
    results = one_d_sweep(base_config, "bid_cost", values = [0.0, 0.05, 0.1])
    for cfg, logs in results:
        plots.windowed_metric(logs["reward"], 10, "Reward")

elif type == "two_d_sweep":
    # results = two_d_sweep(base_config, param_name1, param_name2, values1, values2)
    pass


