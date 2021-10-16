# USE GSAA/60 FOR GOALIE STATS

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
matchupsInput = pd.read_csv('matchups.csv')
curMatchup = 0
homeTeam = matchupsInput.loc[curMatchup]['HomeTeam']
awayTeam = matchupsInput.loc[curMatchup]['AwayTeam']
goalieStats['HomeGoalie'] = matchupsInput.loc[curMatchup]['HomeGoalie']
goalieStats['AwayGoalie'] = matchupsInput.loc[curMatchup]['AwayGoalie']
daysRest_H = matchupsInput.loc[curMatchup]['HomeDaysRest']
daysRest_A = matchupsInput.loc[curMatchup]['AwayDaysRest']

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
    if teamName == 'New York Islanders': replaceNames['Sebastian Aho'] = 'Sebastian Aho_NYI'
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
    goalieStats[curSituation[0] + curSituation[1] + 'A'] = goalieStats[adjSituation].loc[goalieStats['HomeGoalie']][curSituation[0] + 'GSAA/60']
    goalieStats[curSituation[0] + curSituation[1] + 'H'] = goalieStats[adjSituation].loc[goalieStats['AwayGoalie']][curSituation[0] + 'GSAA/60']

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
for curStat in SC_prob_compiled.keys():
    for dangeri in ['HD','MD','LD']:
        # Subtraction in following equation, because if goalie stat if below average you'd increase scoring probability
        if (dangeri in curStat) and (curStat.endswith('H')):
            SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (goalieStats['EV'].loc[goalieStats['AwayGoalie']][dangeri + 'SV%']*SC_prob_compiled[curStat])
        elif (dangeri in curStat) and (curStat.endswith('A')):
            SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (goalieStats['EV'].loc[goalieStats['HomeGoalie']][dangeri + 'SV%']*SC_prob_compiled[curStat])


# ADJUST SCORING PROBABILITY BASED ON OPPOSING GOALIE SV% AND REST ADVANTAGE
for curStat in SC_prob_compiled.keys():
    # Adjust based on goal differential of rest advantage
    if curStat.endswith('H'): # Home Team Stat
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] + (SC_prob_compiled[curStat]*restAdj_Goals)
    else:
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (SC_prob_compiled[curStat]*restAdj_Goals)


# SIMULATE OUTCOMES OF EACH SITUATION
numSims = 10000
compiled_outcomes_H, compiled_outcomes_A = sog_outcome_simulator.simulate_sog(numSims,SC_pred_compiled,SC_prob_compiled)

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
homeOdds = matchupsInput.loc[curMatchup]['HomeOdds']
awayOdds = matchupsInput.loc[curMatchup]['AwayOdds']
kellyMultiplier = 1
# Convert odds to Decimal Odds
def convertOdds(odds):
    if odds < 0: odds = 1-(100/odds)
    else: odds = (odds/100)+1
    return odds

homeOdds = convertOdds(homeOdds)
awayOdds = convertOdds(awayOdds)

kellyValue_H = ((homeOdds - 1) * (winProb_H_notie/100) - (1 - (winProb_H_notie/100))) / (homeOdds - 1) * kellyMultiplier
kellyValue_A = ((awayOdds - 1) * (winProb_A_notie/100) - (1 - (winProb_A_notie/100))) / (awayOdds - 1) * kellyMultiplier

print('Kelly Values')
print(f"{awayTeam} = {round(kellyValue_A*100,2)}%")
print(f"{homeTeam} = {round(kellyValue_H*100,2)}%")
print()

#PRINT LINEUPS

assorted_minor_functions.printProjLineups(homeTeam,awayTeam,names)





