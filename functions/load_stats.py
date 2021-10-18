# LOAD STATISTICS FILES

# Add paths of additional scripts
import sys
sys.path.append('./functions')


import itertools
import pandas as pd
import numpy as np
import replace_missing_data


def loadStats():

    teamStats = dict()
    for curSituation in list(itertools.product(['H','A'],['EV','PP','PK'])):
        teamStats[curSituation[1] + '_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Rates_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
        teamStats[curSituation[1] + '_cnts_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Counts_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
    
    # GET BASELINE CORSI AND GOALS NUMBERS TO ADJUST PLAYER VALUES
    baseline_SC = dict()
    for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['CF','CA','GF','GA'])):
        curBaselineDF = pd.read_csv('input/2019_2021_TeamStats_Calcpermin_' + curSituation[1] + '.csv',encoding= 'unicode_escape').set_index('Team',drop=True)
        baseline_SC[curSituation[0] + curSituation[1] + curSituation[2]] = curBaselineDF.loc['SUM'][curSituation[0] + curSituation[2] + 'permin']
        
        
        #if curSituation[3] == 'off':
        #    baseline_SC[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = curBaselineDF.loc['SUM'][curSituation[0] + 'CFpermin']
        #elif curSituation[3] == 'def':
        #    baseline_SC[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = curBaselineDF.loc['SUM'][curSituation[0] + 'CApermin']
    
    
    
    
    playerStats = dict()
    for curSituation in ['EV','PP','PK']:
        playerStats[curSituation] = pd.read_csv('input/2020_2021_PlayerStats_Rates_OnIce_' + curSituation + '.csv',encoding= 'unicode_escape').set_index('Player',drop=True)
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
    
    playerStats_relative = dict()
    for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['CF','CA','GF','GA'])):
        playerStats_relative[curSituation[1]] = pd.read_csv('input/2020_2021_PlayerStats_Relative_OnIce_' + curSituation[1] + '.csv',encoding= 'unicode_escape').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        playerStats_relative[curSituation[1]] = replace_missing_data.replaceMissingValues(playerStats_relative[curSituation[1]])
        
        # Adjust relative values based on baseline numbers - First make values per minute
        playerStats_relative[curSituation[1]][curSituation[0] + curSituation[2] + 'adjpermin'] = (playerStats_relative[curSituation[1]][curSituation[0] + curSituation[2] + '/60 Rel']/60) + baseline_SC[curSituation[0] + curSituation[1] + curSituation[2]]
        
        
        
        # Calculate New Columns
        def dangerSCProb_off_relative(x):
            try:
                return x[dangeri + 'GF/60 Rel']/x[dangeri + 'CF/60 Rel']
            except:
                return 0
        def dangerSCProb_def_relative(x):
            #try:
            return x[dangeri + 'GA/60 Rel']/x[dangeri + 'CA/60 Rel']
            #except:
            #    return 0
        #for dangeri in ['HD','MD','LD']:
            # Offensive Stats
            #playerStats_relative[curSituation][dangeri + '_SC_prob_off'] = playerStats_relative[curSituation].apply(lambda x: dangerSCProb_off(x), axis=1)
            #playerStats_relative[curSituation][dangeri + '_SC_prob_off'] = playerStats_relative[curSituation][dangeri + 'GF/60 Rel']/playerStats_relative[curSituation][dangeri + 'CF/60 Rel']
            #playerStats_relative[curSituation][dangeri + '_SC_permin_off'] = playerStats_relative[curSituation].apply(lambda x: x[dangeri + 'CF/60 Rel']/60, axis=1)
            
            
            # Defensive Stats
            #playerStats_relative[curSituation][dangeri + '_SC_prob_def'] = playerStats_relative[curSituation].apply(lambda x: dangerSCProb_def(x), axis=1)
            #playerStats_relative[curSituation][dangeri + '_SC_prob_def'] = playerStats_relative[curSituation][dangeri + 'GA/60 Rel']/playerStats_relative[curSituation][dangeri + 'CA/60 Rel']
            #playerStats_relative[curSituation][dangeri + '_SC_permin_def'] = playerStats_relative[curSituation].apply(lambda x: x[dangeri + 'CA/60 Rel']/60, axis=1)
    
        
        
    goalieStats = dict()
    for curSituation in ['EV','PP','PK']:
        goalieStats[curSituation] = pd.read_csv('input/2019_2021_GoalieStats_Rates_' + curSituation + '.csv').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        goalieStats[curSituation] = replace_missing_data.replaceMissingValues_goalies(goalieStats[curSituation])
        # Calculate Columns Relative to Average
        for curCol in ['HDSV%','MDSV%','LDSV%']:
            curAVG = np.mean(goalieStats[curSituation][curCol])
            goalieStats[curSituation][curCol] = goalieStats[curSituation][curCol].apply(lambda x: (x-curAVG)/curAVG)
    
    return teamStats, playerStats, goalieStats, playerStats_relative
