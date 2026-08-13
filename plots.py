import numpy as np
import matplotlib.pyplot as plt


def windowed_metric(data, window_size, y_axis):

    x_intervals = [i for i in range(0, len(data), window_size)] # note that the x for an interval will only show up as the starting value (purely display)

    windowed_data = []
    for i in range(0, len(data), window_size):
        windowed_data.append(np.mean(data[i:i+window_size]))

    plt.plot(x_intervals, windowed_data)
    plt.xlabel("Step Range")
    plt.ylabel(y_axis)
    plt.show()
    plt.close()


def bid_ratio_round(bids, vals, num_rounds, window_size): # window_size is also arbitrary

    by_round_bids = [bids[i::num_rounds] for i in range(num_rounds)] # sorts into a list of num_rounds lists, one per round
    by_round_vals = [vals[i::num_rounds] for i in range(num_rounds)]

    for i in range(num_rounds):

        step_range = [k for k in range(0, len(by_round_bids[i]), window_size)]

        windowed_bids = []
        windowed_vals = []
        for j in range(0, len(by_round_bids[i]), window_size):
            windowed_bids.append(np.mean(by_round_bids[i][j:j+window_size]))
            windowed_vals.append(np.mean(by_round_vals[i][j:j+window_size]))

        ratios = [b / v for b, v in zip(windowed_bids, windowed_vals)]
        plt.plot(step_range, ratios, label = f"Round {i + 1}")

    plt.xlabel("Step Range")
    plt.ylabel("Bid Ratio")
    plt.legend()
    plt.show()
    plt.close()
