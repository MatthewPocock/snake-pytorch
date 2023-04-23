import matplotlib.pyplot as plt
from IPython import display
import pandas as pd

plt.ion()

def plot(scores, mean_scores):
    rolling_mean = pd.Series(scores).rolling(window=50).mean()
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.scatter(range(len(scores)), scores, alpha=.3)
    plt.plot(rolling_mean, color='m')
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if len(rolling_mean) > 0:
        plt.text(len(rolling_mean)-1, rolling_mean.iloc[-1], str(rolling_mean.iloc[-1]))
    plt.show(block=False)
    plt.pause(.1)
