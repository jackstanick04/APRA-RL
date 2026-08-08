import numpy as np
import matplotlib.pyplot as plt

# step ranges should be parameterized
# i think ideally we have a plot function within the driver class that appropriately calls all of these methods down the line
# can also maybe consolidate these all into one parameterized function realistically? but may be hard to work with the labels and may not be worth it

def plot_loss(losses): # can parameterize this

    step_range = [i for i in range(0, len(losses), 1500)]

    windowed_losses = []
    for i in range(0, len(losses), 1500): 
        windowed_losses.append(np.mean(losses[i:i+1500])) # average over blocks of size 1500 here

    plt.plot(step_range, windowed_losses) 
    plt.xlabel("Step Range")
    plt.ylabel("Average Loss")
    plt.show()
    plt.close()

def bid_ratio_round(bids, vals, num_rounds):

    by_round_bids = [bids[i::num_rounds] for i in range(num_rounds)] # look up syntax, but sorts into a list of 5 lists (one per round) beautifully
    by_round_vals = [vals[i::num_rounds] for i in range(num_rounds)]

    for i in range(num_rounds):
        
        step_range = [k for k in range(0, len(by_round_bids[i]), 100)]

        windowed_bids = []
        windowed_vals = []
        for j in range(0, len(by_round_bids[i]), 100): 
            windowed_bids.append(np.mean(by_round_bids[i][j:j+100]))
            windowed_vals.append(np.mean(by_round_vals[i][j:j+100]))

        ratios = [b / v for b, v in zip(windowed_bids, windowed_vals)]
        plt.plot(step_range, ratios, label = f"Round {i + 1}") 
        
    plt.xlabel("Step Range")
    plt.ylabel("Bid Ratio")
    plt.legend()
    plt.show()
    plt.close()

def plot_rewards(rewards):

    step_range = [i for i in range(0, len(rewards), 100)]

    windowed_rewards = []
    for i in range(0, len(rewards), 100): 
        windowed_rewards.append(np.mean(rewards[i:i+100])) # average over blocks of size 100 here

    plt.plot(step_range, windowed_rewards) 
    plt.xlabel("Step Range")
    plt.ylabel("Average Reward")
    plt.show()
    plt.close()

def plot_win_rate(wins):

    step_range = [i for i in range(0, len(wins), 100)]

    windowed_wins = []
    for i in range(0, len(wins), 100): 
        windowed_wins.append(np.mean(wins[i:i+100])) # average over blocks of size 100 here

    plt.plot(step_range, windowed_wins) 
    plt.xlabel("Step Range")
    plt.ylabel("Win Rate")
    plt.show()
    plt.close()

def plot_revenue(revenues):

    step_range = [i for i in range(0, len(revenues), 100)]

    windowed_revenues = []
    for i in range(0, len(revenues), 100): 
        windowed_revenues.append(np.mean(revenues[i:i+100])) # average over blocks of size 100 here

    plt.plot(step_range, windowed_revenues) 
    plt.xlabel("Step Range")
    plt.ylabel("Average Revenues")
    plt.show()
    plt.close()


