# Add paths of additional scripts
import sys
sys.path.append('./functions')



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools

import assorted_plots
import assorted_minor_functions
import sog_outcome_simulator
import load_stats


# LOAD STATISTICS FILES
teamStats, goalieStats = load_stats.loadStats()

# INPUT
homeTeam = 'Tampa Bay Lightning'
awayTeam = 'Pittsburgh Penguins'
goalieStats['HomeGoalie'] = 'Andrei Vasilevskiy'
goalieStats['AwayGoalie'] = 'Tristan Jarry'
daysRest_H = 0
daysRest_A = 0

# HOME TEAM ADJUSTED WIN AND GOAL PROBABILITIES BASED ON REST ADVANTAGE
restAdj_WP, restAdj_Goals = assorted_minor_functions.restAdvCalc(daysRest_H,daysRest_A)

# Assign Stats of Current Goalies - Mark Home Goalie as Away, and Away goalie as Home to correspond to ooposing skaters
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    goalieStats[curSituation[0] + curSituation[1] + 'A'] = goalieStats[curSituation[1]].loc[goalieStats['HomeGoalie']][curSituation[0] + 'SV%']
    goalieStats[curSituation[0] + curSituation[1] + 'H'] = goalieStats[curSituation[1]].loc[goalieStats['AwayGoalie']][curSituation[0] + 'SV%']


# CALCULATE PROJECTED SCORING CHANCES AND SCORING PROBABILITY IN EACH SITUATION
def SCNumAndProb(df_curTeam,df_oppTeam,stat,curTeam,oppTeam):
    # Projected scoring chances based on average of scoring chances for and those allowed by opponent
    SC_num = (df_curTeam.loc[curTeam][stat + 'SF/60'] + df_oppTeam.loc[oppTeam][stat + 'SA/60'])/2
    # Scoring probability on scoring chances = goals divided by scoring chances
    SC_prob = ((df_curTeam.loc[curTeam][stat + 'GF/60'] + df_oppTeam.loc[oppTeam][stat + 'GA/60'])/2)/SC_num
    return SC_num/60, SC_prob

SC_cnts = dict()
SC_prob = dict()

for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    SC_cnts[curSituation[0] + curSituation[1] + 'H'], SC_prob[curSituation[0] + curSituation[1] + 'H'] = SCNumAndProb(teamStats[curSituation[1] + '_H'],teamStats[curSituation[1] + '_A'],curSituation[0],homeTeam,awayTeam)
    SC_cnts[curSituation[0] + curSituation[1] + 'A'], SC_prob[curSituation[0] + curSituation[1] + 'A'] = SCNumAndProb(teamStats[curSituation[1] + '_A'],teamStats[curSituation[1] + '_H'],curSituation[0],awayTeam,homeTeam)

# ADJUST SCORING PROBABILITY BASED ON OPPOSING GOALIE SV% AND REST ADVANTAGE
for curStat in SC_prob.keys():
    SC_prob[curStat] = (SC_prob[curStat] + (1-float(goalieStats[curStat])))/2
    # Adjust based on goal differential of rest advantage
    if curStat.endswith('H'): # Home Team Stat
        SC_prob[curStat] = SC_prob[curStat] + (SC_prob[curStat]*restAdj_Goals)
    else:
        SC_prob[curStat] = SC_prob[curStat] - (SC_prob[curStat]*restAdj_Goals)



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


# ADJUST SCORING CHANCE NUMBERS BASED ON SITUATIONAL MINUTES PREDICTIONS
for curStat in SC_cnts.keys():
    if 'EV' in curStat:
        SC_cnts[curStat] = round(SC_cnts[curStat]*EV_TOI_pred)
    elif ('PPH' in curStat) or ('PKA' in curStat):
        SC_cnts[curStat] = round(SC_cnts[curStat]*PP_TOI_pred_H)
    elif ('PPA' in curStat) or ('PKH' in curStat):
        SC_cnts[curStat] = round(SC_cnts[curStat]*PP_TOI_pred_A)

# SIMULATE OUTCOMES OF EACH SITUATION
numSims = 10000
compiled_outcomes_H, compiled_outcomes_A = sog_outcome_simulator.simulate_sog(numSims,SC_cnts,SC_prob)

# Plot Predicted Distribution of Goals for Each Team
assorted_plots.plotPredictedTeamGoals(compiled_outcomes_H,compiled_outcomes_A,homeTeam,awayTeam)


# Calculated Win and Tie Probabilities
winProb = [x1 - x2 for (x1, x2) in zip(compiled_outcomes_H, compiled_outcomes_A)]
winProb_H = round((len([x for x in winProb if x > 0])/numSims)*100,2)
winProb_A = round((len([x for x in winProb if x < 0])/numSims)*100,2)
winProb_T = round((len([x for x in winProb if x == 0])/numSims)*100,2)
# Adjust Probabilities based on Rest Adjustment
winProb_H = winProb_H + restAdj_WP
winProb_A = winProb_A - restAdj_WP

print(f"{awayTeam} = {winProb_A}%")
print(f"Tie = {winProb_T}%")
print(f"{homeTeam} = {winProb_H}%")






