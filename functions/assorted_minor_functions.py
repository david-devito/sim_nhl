
import itertools
import requests
from bs4 import BeautifulSoup
import re

# CALCULATE HOME TEAM REST ADVANTAGE
def restAdvCalc(daysRest_H,daysRest_A):
    # Array of change in Win Prob% based on Rest Advantage - source: https://www.tsn.ca/yost-rest-makes-a-major-difference-in-nhl-performance-1.120073
    #restDiff_WP = [-3, -0.4, -1.1, 0, 3.6, 3, -3.7]
    #restDiff_Goals = [-0.037, -0.014, -0.001, 0, 0.014, 0.02, -0.003]
    restDiff_WP = [2.3, 4.9, 4.2, 5.3, 8.9, 8.3, 1.6]
    restDiff_Goals = [-0.017, 0.006, 0.019, 0.02, 0.034, 0.04, 0.017]
    restDiff = daysRest_H - daysRest_A
    
    if restDiff >= 3: restDiff = 3
    elif restDiff <= -3: restDiff = -3
    
    restAdj_WP = restDiff_WP[restDiff + 3] # Add 3 to get proper position in restDiff_WP Array
    restAdj_Goals = restDiff_Goals[restDiff + 3] # Add 3 to get proper position in restDiff_WP Array
    
    return restAdj_WP,restAdj_Goals


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


def printProjLineups(homeTeam,awayTeam,names):
    
    def printLine(team,numArray,names):
        try:
            print(f"{names[team][numArray[0]]} - {names[team][numArray[1]]} - {names[team][numArray[2]]}")
        except:
            print(f"{names[team][numArray[0]]} - {names[team][numArray[1]]}")
    
    numArray = [[0,1,2],[3,4,5],[6,7,8],[9,10,11],[12,13],[14,15],[16,17]]
    
    for teamPlayer in list(itertools.product(['A','H'],numArray)):
        if teamPlayer[1] == numArray[0]:
            if teamPlayer[0] == 'A': print(awayTeam)
            elif teamPlayer[0] == 'H': print(homeTeam)
        printLine(teamPlayer[0],teamPlayer[1],names)
        if teamPlayer[1] == numArray[-1]: print()
    

def kellyCalculation(matchupsInput,curMatchup,winProb_H_notie,winProb_A_notie,awayTeam,homeTeam):
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