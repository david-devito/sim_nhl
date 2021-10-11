# LOAD STATISTICS FILES

import itertools
import pandas as pd

def loadStats():

    teamStats = dict()
    for curSituation in list(itertools.product(['H','A'],['EV','PP','PK'])):
        teamStats[curSituation[1] + '_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Rates_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
        teamStats[curSituation[1] + '_cnts_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Counts_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
    
    goalieStats = dict()
    for curSituation in ['EV','PP','PK']:
        goalieStats[curSituation] = pd.read_csv('input/2020_2021_GoalieStats_Rates_' + curSituation + '.csv').set_index('Player',drop=True)
    
    return teamStats, goalieStats
