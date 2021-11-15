
import itertools
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import pandas as pd

# CALCULATE HOME TEAM REST ADVANTAGE
def restAdvCalc(daysRest_H,daysRest_A):
    # Array of change in Win Prob% based on Rest Advantage - source: https://www.tsn.ca/yost-rest-makes-a-major-difference-in-nhl-performance-1.120073
    restDiff_WP = [2.3, 4.9, 4.2, 5.3, 8.9, 8.3, 1.6]
    restDiff_Goals = [-0.017, 0.006, 0.019, 0.02, 0.034, 0.04, 0.017]
    restDiff = daysRest_H - daysRest_A
    # Make max difference 3 as that is max difference that was included in the analysis
    if restDiff >= 3: restDiff = 3
    elif restDiff <= -3: restDiff = -3
    
    restAdj_WP = restDiff_WP[restDiff + 3] # Add 3 to get proper position in restDiff_WP Array
    restAdj_Goals = restDiff_Goals[restDiff + 3] # Add 3 to get proper position in restDiff_WP Array
    
    return restAdj_WP,restAdj_Goals

# SCRAPE CURRENT LINEUP FROM DAILYFACEOFF
def getLineup(teamName):
    url = "https://www.dailyfaceoff.com/teams/" + teamName + "/line-combinations/"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.content, "lxml")
    name_data = soup.find_all("a", {"class": "player-link"})
    # Create list of player names
    names = []
    for i in range(0,len(name_data)): names.append(name_data[i].img["alt"])
    
    # Replace names that appear differently from stats CSV files
    replaceNames = {'Tim Stützle':'Tim Stuetzle', 'Pierre-Édouard Bellemare':'Pierre-Edouard Bellemare','Will Carrier':'William Carrier','Dmitri Orlov':'Dmitry Orlov',
                    'Patrick Maroon':'Pat Maroon','T.J. Brodie':'TJ Brodie','Mitch Marner':'Mitchell Marner','Michael Matheson':'Mike Matheson','Dan DeKeyser':'Danny DeKeyser',
                    'JT Compher':'J.T. Compher','Matthew Nieto':'Matt Nieto','Chris Tanev':'Christopher Tanev','Mathew Dumba':'Matt Dumba','Maxime Comtois':'Max Comtois',
                    'Matt Benning':'Matthew Benning'}
    # 2 Sebastian Aho's so Replace NYI version
    if teamName == 'New York Islanders': replaceNames['Sebastian Aho'] = 'Sebastian Aho_NYI'
    names = [replaceNames[x] if x in replaceNames.keys() else x for x in names]
    
    return names

# PRINT PROJECTED LINEUPS
def printProjLineups(homeTeam,awayTeam,names):
    
    def printLine(team,numArray,names):
        # Separating printing of forward lines vs. defense pairs
        try:
            print(f"{names[team][numArray[0]]} - {names[team][numArray[1]]} - {names[team][numArray[2]]}")
        except:
            print(f"{names[team][numArray[0]]} - {names[team][numArray[1]]}")
    
    numArray = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13],[14,15],[16,17]]
    
    for teamPlayer in list(itertools.product(['A','H'],numArray)):
        # Print Team Name
        if teamPlayer[1] == numArray[0]:
            if teamPlayer[0] == 'A': print(awayTeam)
            elif teamPlayer[0] == 'H': print(homeTeam)
        printLine(teamPlayer[0],teamPlayer[1],names)
        if teamPlayer[1] == numArray[-1]: print()
    
# CALCULATE KELLY VALUE
def kellyCalculation(prob1_notie,prob2_notie,odds1,odds2,printName1,printName2):
    kellyMultiplier = 1
    # Convert odds to Decimal Odds
    def convertOdds(odds):
        if odds < 0: odds = 1-(100/odds)
        else: odds = (odds/100)+1
        return odds
    
    odds1 = convertOdds(odds1)
    odds2 = convertOdds(odds2)
    
    # Calculate Kelly Value and print result
    kellyValue_1 = ((odds1 - 1) * (prob1_notie/100) - (1 - (prob1_notie/100))) / (odds1 - 1) * kellyMultiplier
    kellyValue_2 = ((odds2 - 1) * (prob2_notie/100) - (1 - (prob2_notie/100))) / (odds2 - 1) * kellyMultiplier
    
    print('Kelly Values')
    print(f"{printName2} = {round(kellyValue_2*100,2)}%")
    print(f"{printName1} = {round(kellyValue_1*100,2)}%")
    print()
    
    
# Get TOI% For all skaters - Doesn't yet take into account 4 on 4
def TOIPercent(curSituation,names):
    TOIPercent = pd.read_csv('input/2020_2021_TOIPercent_' + curSituation + '.csv')
    # Remove % sign from values
    TOIPercent['TOI%'] = TOIPercent['TOI%'].apply(lambda x: float(x[:-1]))
    # Get current Median % to fill players with no data
    curMedian_F = np.nanmedian(TOIPercent[TOIPercent['Position'] == 'F']['TOI%']) - np.std(TOIPercent[TOIPercent['Position'] == 'F']['TOI%'])
    curMedian_D = np.nanmedian(TOIPercent[TOIPercent['Position'] == 'D']['TOI%']) - np.std(TOIPercent[TOIPercent['Position'] == 'D']['TOI%'])
    replaceNameList = [['Tim StÜtzle','Tim Stuetzle'],['Pierre-luc Dubois','Pierre-Luc Dubois'],['Dylan Demelo','Dylan DeMelo'],['Mitch Marner','Mitchell Marner']]
    for replacei in replaceNameList:
        TOIPercent = TOIPercent.replace(replacei[0],replacei[1])
    TOIPercent.set_index('Player',inplace=True)
    
    def getCurTOI(curName,TOIPercent,curMedian):
        try:
            return TOIPercent.loc[curName]['TOI%']
        except:
            return curMedian
    
    player_TOIPerc_F = [getCurTOI(x,TOIPercent,curMedian_F) for x in names[0:12]]
    player_TOIPerc_F = [x/np.sum(player_TOIPerc_F) for x in player_TOIPerc_F]
    player_TOIPerc_D = [getCurTOI(x,TOIPercent,curMedian_D) for x in names[12:18]]
    player_TOIPerc_D = [x/np.sum(player_TOIPerc_D) for x in player_TOIPerc_D]
    
    
    return player_TOIPerc_F, player_TOIPerc_D

# CALCULATE PROPORTION OF EVEN STRENGTH TIME THAT EACH LINE AND DPAIR IS ON THE ICE
def get_EV_TOI_dist(names):
    
    player_TOIPerc_F_H, player_TOIPerc_D_H = TOIPercent('EV',names['H'])
    player_TOIPerc_F_A, player_TOIPerc_D_A = TOIPercent('EV',names['A'])
    
    # Calculate Proportion of even strength time that each line is on the ice
    def propTOIByLine(perTOI,numPlayers):
        TOIByLine = [perTOI[i:i + numPlayers] for i in range(0, len(perTOI), numPlayers)]
        avg_TOIBYLine = [np.mean(x) for x in TOIByLine]
        prop_TOIByLine = [x/(np.sum(avg_TOIBYLine)) for x in avg_TOIBYLine]
        return prop_TOIByLine
    
    propTOIEVH_F = propTOIByLine(player_TOIPerc_F_H,3)
    propTOIEVA_F = propTOIByLine(player_TOIPerc_F_A,3)
    propTOIEVH_D = propTOIByLine(player_TOIPerc_D_H,2)
    propTOIEVA_D = propTOIByLine(player_TOIPerc_D_A,2)
    
    return propTOIEVH_F, propTOIEVA_F, propTOIEVH_D, propTOIEVA_D

def TotalTOIBySituation(teamStats,homeTeam,awayTeam):

    def TeamLevelTOIBySituation(df,team,HorA):
        # Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
        total_TOI = df['EV_cnts' + HorA].loc[team]['TOI'] + df['PP_cnts' + HorA].loc[team]['TOI'] + df['PK_cnts' + HorA].loc[team]['TOI']
        PP_TOI = (df['PP_cnts' + HorA].loc[team]['TOI']/total_TOI)*60
        PK_TOI = (df['PK_cnts' + HorA].loc[team]['TOI']/total_TOI)*60
        return total_TOI, PP_TOI, PK_TOI
    
    total_TOI_H, PP_TOI_H, PK_TOI_H = TeamLevelTOIBySituation(teamStats,homeTeam,'_H')
    total_TOI_A, PP_TOI_A, PK_TOI_A = TeamLevelTOIBySituation(teamStats,awayTeam,'_A')
    
    PP_TOI_pred_H = (PP_TOI_H + PK_TOI_A)/2 # Average of Home PP's and Away PK's
    PP_TOI_pred_A = (PP_TOI_A + PK_TOI_H)/2 # Average of Away PP's and Home PK's
    EV_TOI_pred = 60 - PP_TOI_pred_H - PP_TOI_pred_A
    
    return EV_TOI_pred, PP_TOI_pred_H, PP_TOI_pred_A
    
def calcWinProb(compiled_outcomes_H,compiled_outcomes_A,numSims,awayTeam,homeTeam):
    # Calculated Win and Tie Probabilities
    winProb = [x1 - x2 for (x1, x2) in zip(compiled_outcomes_H, compiled_outcomes_A)]
    winProb_H = round((len([x for x in winProb if x > 0])/numSims)*100,2)
    winProb_A = round((len([x for x in winProb if x < 0])/numSims)*100,2)
    winProb_T = round((len([x for x in winProb if x == 0])/numSims)*100,2)
    
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
    
    return winProb_H_notie, winProb_A_notie
    








