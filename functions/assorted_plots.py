# Plot Distribution of Predicted Goals for Each Team

import numpy as np
import matplotlib.pyplot as plt


def plotPredictedTeamGoals(compiled_outcomes_H,compiled_outcomes_A,homeTeam,awayTeam):


    labels, counts = np.unique(compiled_outcomes_H, return_counts=True)
    plt.bar(labels-0.2, counts, align='center',width=0.4,alpha=0.5,facecolor='red',edgecolor='black')
    plt.gca().set_xticks(labels)
    labels, counts = np.unique(compiled_outcomes_A, return_counts=True)
    plt.bar(labels+0.2, counts, align='center',width=0.4,alpha=0.5,facecolor='green',edgecolor='black')
    colors = {homeTeam:'red', awayTeam:'green'}         
    labels = list(colors.keys())
    handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
    plt.legend(handles, labels)
    
    plt.axvline(np.mean(compiled_outcomes_H), color='red', linewidth=3)
    plt.axvline(np.mean(compiled_outcomes_A), color='green', linewidth=3)
    
    
    plt.show()