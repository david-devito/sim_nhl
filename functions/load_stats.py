# LOAD STATISTICS FILES

# Add paths of additional scripts
import sys
sys.path.append('./functions')


import itertools
import pandas as pd
import replace_missing_data

def loadStats():

    teamStats = dict()
    for curSituation in list(itertools.product(['H','A'],['EV','PP','PK'])):
        teamStats[curSituation[1] + '_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Rates_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
        teamStats[curSituation[1] + '_cnts_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Counts_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
    
    goalieStats = dict()
    for curSituation in ['EV','PP','PK']:
        goalieStats[curSituation] = pd.read_csv('input/2019_2021_GoalieStats_Rates_' + curSituation + '.csv').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        goalieStats[curSituation] = replace_missing_data.replaceMissingValues(goalieStats[curSituation])
    
    return teamStats, goalieStats
