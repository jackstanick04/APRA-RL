import numpy as np
import matplotlib.pyplot as plt

def plot_loss(losses): # can parameterize this

    step_range = [i for i in range(0, len(losses), 1500)]

    windowed_losses = []
    for i in range(0, len(losses), 1500): 
        windowed_losses.append(np.mean(losses[i:i+1500])) # average over blocks of size 100 here

    plt.plot(step_range, windowed_losses) 
    plt.xlabel("Step Range")
    plt.ylabel("Average Loss")
    plt.show()
    plt.close()


