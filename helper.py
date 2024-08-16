### modified from: https://github.com/patrickloeber/snake-ai-pytorch

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

import matplotlib.pyplot as plt

def plot(scores, mean_scores, hyperparameters=None, save_path=None, show_final=False):
    plt.ion()  # Turn on interactive mode
    plt.figure(1)
    plt.clf()  # Clear the current figure

    plt.title('Training Progress')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores, label='Score per Game')
    plt.plot(mean_scores, label='Average Score (Last 100 Games)')
    plt.legend()

    # Only show the plot without hyperparameters during training
    if show_final and hyperparameters:
        plt.text(0.5, 0.5, hyperparameters, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    plt.pause(0.1)  # Pause to allow the plot to update

    # Save the final plot if requested
    if save_path and show_final:
        plt.savefig(save_path)

    if show_final:
        plt.ioff()  # Turn off interactive mode
        plt.show()  # Show the final plot


    
