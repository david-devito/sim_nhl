# Add paths of additional scripts
import sys
sys.path.append('./functions')


import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

import assorted_plots
import assorted_minor_functions
import sog_outcome_simulator
import load_stats
import assign_stats


# LOAD STATISTICS FILES
# Replace NAN values or values for players with fewer than 30 mins played

teamStats, playerStats, goalieStats, playerStats_relative = load_stats.loadStats()
'''
# INPUT
matchupsInput = pd.read_csv('matchups.csv')
curMatchup = 1
homeTeam = matchupsInput.loc[curMatchup]['HomeTeam']
awayTeam = matchupsInput.loc[curMatchup]['AwayTeam']
goalieStats['HomeGoalie'] = matchupsInput.loc[curMatchup]['HomeGoalie']
goalieStats['AwayGoalie'] = matchupsInput.loc[curMatchup]['AwayGoalie']
daysRest_H = matchupsInput.loc[curMatchup]['HomeDaysRest']
daysRest_A = matchupsInput.loc[curMatchup]['AwayDaysRest']

# HOME TEAM ADJUSTED WIN AND GOAL PROBABILITIES BASED ON REST ADVANTAGE
restAdj_WP, restAdj_Goals = assorted_minor_functions.restAdvCalc(daysRest_H,daysRest_A)


## GET LINEUP INFORMATION
names = dict()
names['H'] = assorted_minor_functions.getLineup(homeTeam.replace(' ','-').lower())
names['A'] = assorted_minor_functions.getLineup(awayTeam.replace(' ','-').lower())


# Assign Stats of Current Goalies - Mark Home Goalie as Away, and Away goalie as Home to correspond to ooposing skaters
goalieStats = assign_stats.assignGoalieStats(goalieStats)


# MINUTES DISTRIBUTION
def calcTeamTOIBySituation(df,team,HorA):
    # Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
    total_TOI = df['EV_cnts' + HorA].loc[team]['TOI'] + df['PP_cnts' + HorA].loc[team]['TOI'] + df['PK_cnts' + HorA].loc[team]['TOI']
    PP_TOI = (df['PP_cnts' + HorA].loc[team]['TOI']/total_TOI)*60
    PK_TOI = (df['PK_cnts' + HorA].loc[team]['TOI']/total_TOI)*60
    return total_TOI, PP_TOI, PK_TOI
    
total_TOI_H, PP_TOI_H, PK_TOI_H = calcTeamTOIBySituation(teamStats,homeTeam,'_H')
total_TOI_A, PP_TOI_A, PK_TOI_A = calcTeamTOIBySituation(teamStats,awayTeam,'_A')

# Predicted Special Teams TOI
PP_TOI_pred_H = (PP_TOI_H + PK_TOI_A)/2
PP_TOI_pred_A = (PP_TOI_A + PK_TOI_H)/2
EV_TOI_pred = 60 - PP_TOI_pred_H - PP_TOI_pred_A

# Get TOI% For all skaters - Doesn't yet take into account 4 on 4
def TOIPercent(curSituation,timeFactor,names):
    TOIPercent = pd.read_csv('input/2020_2021_TOIPercent_' + curSituation + '.csv')
    TOIPercent['TOI%'] = TOIPercent['TOI%'].apply(lambda x: float(x[:-1]))
    # Get current Median % to fill players with no data
    curMedian = np.nanmedian(TOIPercent['TOI%'])
    replaceNameList = [['Tim StÜtzle','Tim Stuetzle'],['Pierre-luc Dubois','Pierre-Luc Dubois'],['Dylan Demelo','Dylan DeMelo']]
    for replacei in replaceNameList:
        TOIPercent = TOIPercent.replace(replacei[0],replacei[1])
    TOIPercent.set_index('Player',inplace=True)
    
    def getCurTOI(curName,TOIPercent,curMedian):
        try:
            return TOIPercent.loc[curName]['TOI%']
        except:
            return curMedian
    
    player_TOIPerc = [getCurTOI(x,TOIPercent,curMedian) for x in names[0:18]]
    player_TOIPerc_F = [x/np.sum(player_TOIPerc[0:12]) for x in player_TOIPerc[0:12]]
    player_TOIPerc_F = [(x * timeFactor*3) for x in player_TOIPerc_F]
    player_TOIPerc_D = [x/np.sum(player_TOIPerc[12:18]) for x in player_TOIPerc[12:18]]
    player_TOIPerc_D = [(x * timeFactor*2) for x in player_TOIPerc_D]
    
    player_TOIPerc = player_TOIPerc_F + player_TOIPerc_D
    
    return player_TOIPerc


SC_pred = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurPred(x,OffOrDef):
        try:
            return playerStats[curSituation[1]].loc[x][curSituation[0] + '_SC_permin_' + OffOrDef]
        except:
            return np.median(playerStats[curSituation[1]][curSituation[0] + '_SC_permin_' + OffOrDef])
    
    SC_pred[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurPred(x,curSituation[3]) for x in names[curSituation[2]][0:18]]
    SC_pred[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = int((np.sum([a * b for a, b in zip(player_TOIPerc, SC_pred[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))*timeFactor)

SC_pred_compiled = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    SC_pred_compiled[curSituation[0] + curSituation[1] + 'H'] = int((SC_pred[curSituation[0] + curSituation[1] + 'Hoff'] + SC_pred[curSituation[0] + curSituation[1] + 'Adef'])/2)
    SC_pred_compiled[curSituation[0] + curSituation[1] + 'A'] = int((SC_pred[curSituation[0] + curSituation[1] + 'Aoff'] + SC_pred[curSituation[0] + curSituation[1] + 'Hdef'])/2)


# CALCULATE BASELINE SCORING CHANCE NUMBERS
baseline_SC = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    curBaselineDF = pd.read_csv('input/2019_2021_TeamStats_Calcpermin_' + curSituation[1] + '.csv',encoding= 'unicode_escape').set_index('Team',drop=True)
    if curSituation[3] == 'off':
        baseline_SC[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = curBaselineDF.loc['SUM'][curSituation[0] + 'CFpermin']
    elif curSituation[3] == 'def':
        baseline_SC[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = curBaselineDF.loc['SUM'][curSituation[0] + 'CApermin']

# CALCULATE PREDICTED RELATIVE SCORING CHANCES
SC_pred_relative = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurPred(x,OffOrDef):
        try:
            return playerStats_relative[curSituation[1]].loc[x][curSituation[0] + '_SC_permin_' + OffOrDef]
        except:
            return np.median(playerStats_relative[curSituation[1]][curSituation[0] + '_SC_permin_' + OffOrDef])
    
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurPred(x,curSituation[3]) for x in names[curSituation[2]][0:18]]
    # Get weighted scoring chances based on projected mins per player in current situation
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = (np.sum([a * b for a, b in zip(player_TOIPerc, SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))
    

SC_pred_relative_w_baseline = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    SC_pred_relative_w_baseline[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = int((baseline_SC[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] + SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])*timeFactor)


SC_pred_compiled_relative = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    SC_pred_compiled_relative[curSituation[0] + curSituation[1] + 'H'] = int((SC_pred_relative_w_baseline[curSituation[0] + curSituation[1] + 'Hoff'] + SC_pred_relative_w_baseline[curSituation[0] + curSituation[1] + 'Adef'])/2)
    SC_pred_compiled_relative[curSituation[0] + curSituation[1] + 'A'] = int((SC_pred_relative_w_baseline[curSituation[0] + curSituation[1] + 'Aoff'] + SC_pred_relative_w_baseline[curSituation[0] + curSituation[1] + 'Hdef'])/2)





SC_prob = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurProb(x,OffOrDef):
        try:
            return playerStats[curSituation[1]].loc[x][curSituation[0] + '_SC_prob_' + OffOrDef]
        except:
            return np.median(playerStats[curSituation[1]][curSituation[0] + '_SC_prob_' + OffOrDef])
    
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurProb(x,curSituation[3]) for x in names[curSituation[2]][0:18]]
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = (np.sum([a * b for a, b in zip(player_TOIPerc, SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))
    
SC_prob_compiled = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    SC_prob_compiled[curSituation[0] + curSituation[1] + 'H'] = (SC_prob[curSituation[0] + curSituation[1] + 'Hoff'] + SC_prob[curSituation[0] + curSituation[1] + 'Adef'])/2
    SC_prob_compiled[curSituation[0] + curSituation[1] + 'A'] = (SC_prob[curSituation[0] + curSituation[1] + 'Aoff'] + SC_prob[curSituation[0] + curSituation[1] + 'Hdef'])/2





# ADJUST SCORING PROBABILITY BASED ON OPPOSING GOALIE SV%
def adjByGoalieStat(curStat,dangeri,goalieStats,SC_prob_compiled,whichGoalie):
    try:
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (goalieStats['EV'].loc[goalieStats['AwayGoalie']][dangeri + 'SV%']*SC_prob_compiled[curStat])
    except: # If Goalie has not played then use median of stat
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - ((np.median(goalieStats['EV'][dangeri + 'SV%']))*SC_prob_compiled[curStat])
    
for curStat in SC_prob_compiled.keys():
    for dangeri in ['HD','MD','LD']:
        # Subtraction in following equation, because if goalie stat if below average you'd increase scoring probability
        if (dangeri in curStat) and (curStat.endswith('H')):
            adjByGoalieStat(curStat,dangeri,goalieStats,SC_prob_compiled,'AwayGoalie')
        elif (dangeri in curStat) and (curStat.endswith('A')):
            adjByGoalieStat(curStat,dangeri,goalieStats,SC_prob_compiled,'HomeGoalie')
            
# ADJUST SCORING PROBABILITY BASED ON REST ADVANTAGE
for curStat in SC_prob_compiled.keys():
    # Adjust based on goal differential of rest advantage
    if curStat.endswith('H'): # Home Team Stat
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] + (SC_prob_compiled[curStat]*restAdj_Goals)
    else:
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (SC_prob_compiled[curStat]*restAdj_Goals)


# SIMULATE OUTCOMES OF EACH SITUATION
numSims = 20000
compiled_outcomes_H, compiled_outcomes_A = sog_outcome_simulator.simulate_sog(numSims,SC_pred_compiled_relative,SC_prob_compiled)

# Plot Predicted Distribution of Goals for Each Team
assorted_plots.plotPredictedTeamGoals(compiled_outcomes_H,compiled_outcomes_A,homeTeam,awayTeam)


# Calculated Win and Tie Probabilities
winProb = [x1 - x2 for (x1, x2) in zip(compiled_outcomes_H, compiled_outcomes_A)]
winProb_H = round((len([x for x in winProb if x > 0])/numSims)*100,2)
winProb_A = round((len([x for x in winProb if x < 0])/numSims)*100,2)
winProb_T = round((len([x for x in winProb if x == 0])/numSims)*100,2)
# Adjust Probabilities based on Rest Adjustment
#winProb_H = winProb_H + restAdj_WP
#winProb_A = winProb_A - restAdj_WP

print()
print(f"{awayTeam} = {winProb_A}%")
print(f"Tie = {winProb_T}%")
print(f"{homeTeam} = {winProb_H}%")
print()

# Calculated Win Probabilities - Excluding Ties
winProb_H_notie = round(winProb_H + (winProb_H/(winProb_H + winProb_A))*winProb_T,2)
winProb_A_notie = round(winProb_A + (winProb_A/(winProb_H + winProb_A))*winProb_T,2)
print('When No Tie Possible')
print(f"{awayTeam} = {winProb_A_notie}%")
print(f"{homeTeam} = {winProb_H_notie}%")
print()




# Kelly Criterion Formula
assorted_minor_functions.kellyCalculation(matchupsInput,curMatchup,winProb_H_notie,winProb_A_notie,awayTeam,homeTeam)

#PRINT LINEUPS

assorted_minor_functions.printProjLineups(homeTeam,awayTeam,names)


'''


