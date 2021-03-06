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
        teamStats[curSituation[1] + '_cnts_' + curSituation[0]] = pd.read_csv('input/2020_2021_TeamStats_Counts_' + curSituation[1] + '_' + curSituation[0] + '.csv').set_index('Team',drop=True)
    
    # GET BASELINE CORSI AND GOALS NUMBERS TO ADJUST PLAYER VALUES
    baseline_SC = dict()
    for curDanger in ['HD','MD','LD']:
        for curSituation in list(itertools.product(['EV','PP','PK'],['CF','CA','GF','GA'])):
            curBaselineDF = pd.read_csv('input/2019_2021_TeamStats_Calcpermin_' + curSituation[0] + '.csv',encoding= 'unicode_escape').set_index('Team',drop=True)
            baseline_SC[curDanger + curSituation[0] + curSituation[1]] = curBaselineDF.loc['SUM'][curDanger + curSituation[1] + 'permin']
       
    playerStats_relative = dict()
    for curSituation in list(itertools.product(['EV','PP','PK'],['H','A'])):
        playerStats_relative[curSituation[0] + curSituation[1]] = pd.read_csv('input/2020_2021_PlayerStats_Relative_OnIce_' + curSituation[0] + '_' + curSituation[1] + '.csv',encoding= 'unicode_escape').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        playerStats_relative[curSituation[0] + curSituation[1]] = replace_missing_data.replaceMissingValues(playerStats_relative[curSituation[0] + curSituation[1]])
        
        for dangeri in ['HD','MD','LD']:
            
            for curStat in ['CF','CA','GF','GA']:
                # Adjust relative values based on baseline numbers - First make values per minute
                playerStats_relative[curSituation[0] + curSituation[1]][dangeri + curStat + 'adjpermin'] = baseline_SC[dangeri + curSituation[0] + curStat] + (playerStats_relative[curSituation[0] + curSituation[1]][dangeri + curStat + '/60 Rel']/60)
                # Adjust negative values to be 0
                playerStats_relative[curSituation[0] + curSituation[1]][dangeri + curStat + 'adjpermin'] = playerStats_relative[curSituation[0] + curSituation[1]][dangeri + curStat + 'adjpermin'].apply(lambda x: 0 if x < 0 else x)
        
            # Hard code scenarios with extremely low sample sizes as 0's
            # These include goals against while on the PP, and Goals for while on the penalty kill
            if curSituation[0] == 'PP': playerStats_relative[curSituation[0] + curSituation[1]][dangeri + 'GAadjpermin'] = 0
            elif curSituation == 'PK': playerStats_relative[curSituation[0] + curSituation[1]][dangeri + 'GFadjpermin'] = 0
        
            # Calculate Probability of Scoring Based on Corsi and Goals
            def dangerSCProb_off_relative(x):
                try:
                    return x[dangeri + 'GFadjpermin']/x[dangeri + 'CFadjpermin']
                except:
                    return 0
            def dangerSCProb_def_relative(x):
                try:
                    return x[dangeri + 'GAadjpermin']/x[dangeri + 'CAadjpermin']
                except:
                    return 0
            
            playerStats_relative[curSituation[0] + curSituation[1]][dangeri + '_SC_prob_off'] = playerStats_relative[curSituation[0] + curSituation[1]].apply(lambda x: dangerSCProb_off_relative(x), axis=1)
            playerStats_relative[curSituation[0] + curSituation[1]][dangeri + '_SC_prob_def'] = playerStats_relative[curSituation[0] + curSituation[1]].apply(lambda x: dangerSCProb_def_relative(x), axis=1)
    
    # Correct Tim Stuetzle's name. Accent appears different in different dataframes
    def replaceNames_Tim(df,orig,new):
        as_list = df.index.tolist()
        idx = as_list.index(orig)
        as_list[idx] = new
        df.index = as_list
        return df

    
    for homeAway in ['H','A']:
        playerStats_relative['EV' + homeAway] = replaceNames_Tim(playerStats_relative['EV' + homeAway],'Tim St????tzle','Tim Stuetzle')
        playerStats_relative['PP' + homeAway] = replaceNames_Tim(playerStats_relative['PP' + homeAway],'Tim St????tzle','Tim Stuetzle')
        #playerStats_relative['PK' + homeAway] = replaceNames_Tim(playerStats_relative['PK' + homeAway],'Tim St????tzle','Tim Stuetzle')
        playerStats_relative['EV' + homeAway] = replaceNames_Tim(playerStats_relative['EV' + homeAway],'Alexis Lafreni????re','Alexis Lafreniere')
        playerStats_relative['PP' + homeAway] = replaceNames_Tim(playerStats_relative['PP' + homeAway],'Alexis Lafreni????re','Alexis Lafreniere')
        playerStats_relative['PK' + homeAway] = replaceNames_Tim(playerStats_relative['PK' + homeAway],'Alexis Lafreni????re','Alexis Lafreniere')
        
        
    goalieStats = dict()
    for curSituation in ['EV','PP','PK']:
        goalieStats[curSituation] = pd.read_csv('input/2019_2021_GoalieStats_Rates_' + curSituation + '.csv').set_index('Player',drop=True)
        # Replace NAN values and stats for players with less than 30 minutes with the median of each column
        goalieStats[curSituation] = replace_missing_data.replaceMissingValues_goalies(goalieStats[curSituation])
        # Calculate Columns Relative to Average
        for curCol in ['HDSV%','MDSV%','LDSV%']:
            curAVG = np.mean(goalieStats[curSituation][curCol])
            goalieStats[curSituation][curCol] = goalieStats[curSituation][curCol].apply(lambda x: (x-curAVG)/curAVG)
    
    return teamStats, goalieStats, playerStats_relative, baseline_SC
