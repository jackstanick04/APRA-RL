import dataclasses
from driver import run_trial

def one_d_sweep(base_config, param_name, values):

    results = [] # tuples of (config, logs)

    for value in values:
        config = dataclasses.replace(base_config, **{param_name: value})
        logs = run_trial(config)
        results.append((config, logs))

    return results

# same logic as above function
def two_d_sweep(base_config, param_one, param_two, values_one, values_two):

    results = []

    for value_one in values_one:
        for value_two in values_two:

            config = dataclasses.replace(base_config, **{param_one: value_one, param_two: value_two}) 
            logs = run_trial(config)
            results.append((config, logs))

    return results



