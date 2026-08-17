import numpy as np
import matplotlib.pyplot as plt

# NEED MORE GRAPHS TO IMPLEMENT

def _compute_window(data, window_size):
    return [np.mean(data[i:window_size + i]) for i in range(0, len(data), window_size)]

# for non sweep calls
def windowed_metric(logs, window_size, y_axis):
    fig, ax = plt.subplots()
    ax.plot(_compute_window(logs[y_axis], window_size))
    ax.set_xlabel(f"Episode (windowed, size={window_size})")
    ax.set_ylabel(y_axis.replace("_", " ").title())
    ax.set_title(f"{y_axis.replace("_", " ").title()} v. Episode")
    return fig

# does the plotting for both sweep types
def _plot_sweep_line(ax, logs, window_size, y_axis, label):
    ax.plot(_compute_window(logs[y_axis], window_size), label = label)

def windowed_metric_one_sweep(results, param, window_size, y_axis):
    fig, ax = plt.subplots()
    for config, logs in results:
        _plot_sweep_line(ax, logs, window_size, y_axis, str(getattr(config, param)))

    ax.set_xlabel(f"Episode (windowed, size={window_size})")
    ax.set_ylabel(y_axis.replace("_", " ").title())
    ax.set_title(f"{y_axis.replace("_", " ").title()} v. Episode swept, over {param.replace("_", " ").title()}")
    ax.legend(title = param)
    return fig

def windowed_metric_two_sweep(results, param_one, param_two, window_size, y_axis):

    param_one_vals = sorted({getattr(config, param_one) for config, _ in results})
    fig, axes = plt.subplots(1, len(param_one_vals), sharex = True, sharey = True)

    for ax, a_val in zip(axes, param_one_vals): # structure is len(param_a) graphs with len(param_b) lines on them
        param_two_vals_per_one = [(config, logs) for config, logs in results if getattr(config, param_one) == a_val] # set of all config and logs where param_two varies and param_one is constant
        for config, logs in param_two_vals_per_one:
            _plot_sweep_line(ax, logs, window_size, y_axis, str(getattr(config, param_two)))
        ax.set_title(f"{param_one.replace("_", " ").title()}: {a_val}")

    axes[0].legend(title = param_two.replace("_", " ").title())
    axes[0].set_ylabel(y_axis.replace("_", " ").title())

    fig.supxlabel(f"Episode (windowed, size={window_size})")
    fig.suptitle(f"{y_axis.replace('_', ' ').title()} v. Episode swept, over {param_one.replace("_", " ").title()} and {param_two.replace("_", " ").title()}")
    return fig



