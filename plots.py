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

def one_metric_sweep(data, x_axis, y_axis):

    avg_data = [np.mean(val) for val in data.values()]
    plt.plot(list(data.keys()), avg_data)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.show()
    plt.close

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

def bid_val_round_per_reimburse(bids, vals, num_rounds): # bid to val ratio per round, with a new line for each reimbursement rate

    for r in bids.keys():

        sorted_bids = [np.mean(bids[r][i::num_rounds]) for i in range(num_rounds)]
        sorted_vals = [np.mean(vals[r][i::num_rounds]) for i in range(num_rounds)]
        ratios = [b / v for b, v in zip(sorted_bids, sorted_vals)]

        plt.plot(range(1, num_rounds + 1), ratios, label = f"Reimbursement Rate: {r}")

    plt.xlabel("Round")
    plt.ylabel("Bid Ratio")
    plt.legend()
    plt.show()
    plt.close()

def rev_max_bid_per_reimburse(revenues, max_bids):

    avg_revs = {r: np.mean(revenues[r]) for r in list(revenues.keys())}
    avg_max_bids = {r: np.mean(max_bids[r]) for r in list(max_bids.keys())}

    plt.plot(list(avg_revs.keys()), list(avg_revs.values()), label = "Average Revenue")
    plt.plot(list(avg_max_bids.keys()), list(avg_max_bids.values()), label ="Average Max Bid")

    plt.xlabel("Reimbursement Rate")
    plt.ylabel("Value")
    plt.legend()
    plt.show()
    plt.close()





