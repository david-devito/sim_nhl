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


# LOAD STATISTICS FILES
# Replace NAN values or values for players with fewer than 30 mins played

teamStats, playerStats, goalieStats = load_stats.loadStats()

# INPUT
homeTeam = 'Montreal Canadiens'
awayTeam = 'Winnipeg Jets'
goalieStats['HomeGoalie'] = 'Jake Allen'
goalieStats['AwayGoalie'] = 'Connor Hellebuyck'
daysRest_H = 0
daysRest_A = 0

# HOME TEAM ADJUSTED WIN AND GOAL PROBABILITIES BASED ON REST ADVANTAGE
restAdj_WP, restAdj_Goals = assorted_minor_functions.restAdvCalc(daysRest_H,daysRest_A)


## GET LINEUP INFORMATION

def getLineup(teamName):
    # Scrape team line combos from DFO
    url = "https://www.dailyfaceoff.com/teams/" + teamName + "/line-combinations/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "lxml")
    name_data = soup.find_all("a", {"class": "player-link"})
    # Create list of player names
    names = []
    for i in range(0,len(name_data)): names.append(name_data[i].img["alt"])
    replaceNames = {'Tim Stützle':'Tim Stuetzle', 'Pierre-Édouard Bellemare':'Pierre-Edouard Bellemare'}
    names = [replaceNames[x] if x in replaceNames.keys() else x for x in names]
    return names

names = dict()
names['H'] = getLineup(homeTeam.replace(' ','-').lower())
names['A'] = getLineup(awayTeam.replace(' ','-').lower())


# Assign Stats of Current Goalies - Mark Home Goalie as Away, and Away goalie as Home to correspond to ooposing skaters
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    #Adjust PP/PK for fact that on PK when facing opponent's PP, and vice versa
    if curSituation[1] == 'PP': adjSituation = 'PK'
    elif curSituation[1] == 'PK': adjSituation = 'PP'
    else: adjSituation = 'EV'
    goalieStats[curSituation[0] + curSituation[1] + 'A'] = goalieStats[adjSituation].loc[goalieStats['HomeGoalie']][curSituation[0] + 'SV%']
    goalieStats[curSituation[0] + curSituation[1] + 'H'] = goalieStats[adjSituation].loc[goalieStats['AwayGoalie']][curSituation[0] + 'SV%']


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
    if curSituation[1] == 'PP': adjSituation = 'PK'
    elif curSituation[1] == 'PK': adjSituation = 'PP'
    else: adjSituation = 'EV'
    SC_cnts[curSituation[0] + curSituation[1] + 'H'], SC_prob[curSituation[0] + curSituation[1] + 'H'] = SCNumAndProb(teamStats[curSituation[1] + '_H'],teamStats[adjSituation + '_A'],curSituation[0],homeTeam,awayTeam)
    SC_cnts[curSituation[0] + curSituation[1] + 'A'], SC_prob[curSituation[0] + curSituation[1] + 'A'] = SCNumAndProb(teamStats[curSituation[1] + '_A'],teamStats[adjSituation + '_H'],curSituation[0],awayTeam,homeTeam)

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

# Get TOI%_EV For all skaters - Doesn't yet take into account 4 on 4
def TOIPercent(curSituation,curSituation_TOI_Pred,names):
    TOIPercent = pd.read_csv('input/2020_2021_TOIPercent_' + curSituation + '.csv')
    # Get current Median % to fill players with no data
    curMedian = np.nanmedian(TOIPercent['TOI'])
    replaceNameList = [['Tim StÜtzle','Tim Stuetzle'],['Pierre-luc Dubois','Pierre-Luc Dubois'],['Dylan Demelo','Dylan DeMelo']]
    for replacei in replaceNameList:
        TOIPercent = TOIPercent.replace(replacei[0],replacei[1])
    TOIPercent.set_index('Player',inplace=True)
    
    def getCurTOI(curName,TOIPercent,curMedian):
        try:
            return float(TOIPercent.loc[curName]['TOI%'][:-1])
        except:
            return curMedian
    
    player_TOIPerc = [getCurTOI(x,TOIPercent,curMedian) for x in names[0:18]]
    player_TOIPerc = [x/np.sum(player_TOIPerc) for x in player_TOIPerc]

    return player_TOIPerc

SC_pred = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurPred(x):
        try:
            return playerStats[curSituation[1]].loc[x][curSituation[0] + '_SC_permin']
        except:
            return 0
    
    SC_pred[curSituation[0] + curSituation[1] + curSituation[2]] = [getCurPred(x) for x in names[curSituation[2]][0:18]]
    SC_pred[curSituation[0] + curSituation[1] + curSituation[2]] = int(np.sum([a * b for a, b in zip(player_TOIPerc, SC_pred[curSituation[0] + curSituation[1] + curSituation[2]])])*timeFactor)

SC_prob = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'])):
    def getCurProb(x):
        try:
            return playerStats[curSituation[1]].loc[x][curSituation[0] + '_SC_prob']
        except:
            return 0
    
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2]] = [getCurProb(x) for x in names[curSituation[2]][0:18]]
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2]] = np.mean(SC_prob[curSituation[0] + curSituation[1] + curSituation[2]])






'''
# ADJUST SCORING CHANCE NUMBERS BASED ON SITUATIONAL MINUTES PREDICTIONS
for curStat in SC_cnts.keys():
    if 'EV' in curStat:
        SC_cnts[curStat] = round(SC_cnts[curStat]*EV_TOI_pred)
    elif ('PPH' in curStat) or ('PKA' in curStat):
        SC_cnts[curStat] = round(SC_cnts[curStat]*PP_TOI_pred_H)
    elif ('PPA' in curStat) or ('PKH' in curStat):
        SC_cnts[curStat] = round(SC_cnts[curStat]*PP_TOI_pred_A)
'''
# SIMULATE OUTCOMES OF EACH SITUATION
numSims = 10000
compiled_outcomes_H, compiled_outcomes_A = sog_outcome_simulator.simulate_sog(numSims,SC_pred,SC_prob)

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






