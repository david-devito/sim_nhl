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
    
    playerStats = dict()
    for curSituation in ['EV','PP','PK']:
        playerStats[curSituation] = pd.read_csv('input/2020_2021_PlayerStats_Rates_OnIce_' + curSituation + '.csv').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        playerStats[curSituation] = replace_missing_data.replaceMissingValues(playerStats[curSituation])
        # Calculate New Columns
        def dangerSCProb_off(x):
            try:
                return x[dangeri + 'GF/60']/x[dangeri + 'CF/60']
            except:
                return 0
        def dangerSCProb_def(x):
            try:
                return x[dangeri + 'GA/60']/x[dangeri + 'CA/60']
            except:
                return 0
        for dangeri in ['HD','MD','LD']:
            # Offensive Stats
            playerStats[curSituation][dangeri + '_SC_prob_off'] = playerStats[curSituation].apply(lambda x: dangerSCProb_off(x), axis=1)
            playerStats[curSituation][dangeri + '_SC_permin_off'] = playerStats[curSituation].apply(lambda x: x[dangeri + 'CF/60']/60, axis=1)
            # Defensive Stats
            playerStats[curSituation][dangeri + '_SC_prob_def'] = playerStats[curSituation].apply(lambda x: dangerSCProb_def(x), axis=1)
            playerStats[curSituation][dangeri + '_SC_permin_def'] = playerStats[curSituation].apply(lambda x: x[dangeri + 'CA/60']/60, axis=1)
        
        
        
        
        
    goalieStats = dict()
    for curSituation in ['EV','PP','PK']:
        goalieStats[curSituation] = pd.read_csv('input/2019_2021_GoalieStats_Rates_' + curSituation + '.csv').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        goalieStats[curSituation] = replace_missing_data.replaceMissingValues(goalieStats[curSituation])
    
    return teamStats, playerStats, goalieStats
