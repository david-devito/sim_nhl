## SOME OF THE RELATIVE STATS ARE JUST TOO HIGH SO THE PROBABILITY NUMBERS END UP TOO HIGH

## BASELINE STATS NEED TO BE HOME AND AWAY


# Add paths of additional scripts
import sys
sys.path.append('./functions')

import pandas as pd
import numpy as np
import itertools

import assorted_minor_functions
import load_stats
import time


# LOAD STATISTICS FILES
print('LOAD STATS')
teamStats, goalieStats, playerStats_relative, baseline_SC = load_stats.loadStats()

# INPUT
matchupsInput = pd.read_csv('matchups.csv')
curMatchup = 5
homeTeam = matchupsInput.loc[curMatchup]['HomeTeam']
awayTeam = matchupsInput.loc[curMatchup]['AwayTeam']
goalieStats['HomeGoalie'] = matchupsInput.loc[curMatchup]['HomeGoalie']
goalieStats['AwayGoalie'] = matchupsInput.loc[curMatchup]['AwayGoalie']
daysRest_H = int(matchupsInput.loc[curMatchup]['HomeDaysRest'])
daysRest_A = int(matchupsInput.loc[curMatchup]['AwayDaysRest'])
gameOverUnder = matchupsInput.loc[curMatchup]['OverUnderVal']


# HOME TEAM ADJUSTED WIN AND GOAL PROBABILITIES BASED ON REST ADVANTAGE
restAdj_WP, restAdj_Goals = assorted_minor_functions.restAdvCalc(daysRest_H,daysRest_A)

## GET LINEUP INFORMATION
names = dict()
names['H'] = assorted_minor_functions.getLineup(homeTeam.replace(' ','-').lower())
names['A'] = assorted_minor_functions.getLineup(awayTeam.replace(' ','-').lower())


# MINUTES DISTRIBUTION BY TEAM
print('MINUTES CALC')
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
def TOIPercent(curSituation,names):
    TOIPercent = pd.read_csv('input/2020_2021_TOIPercent_' + curSituation + '.csv')
    # Remove % sign from values
    TOIPercent['TOI%'] = TOIPercent['TOI%'].apply(lambda x: float(x[:-1]))
    # Get current Median % to fill players with no data
    curMedian_F = np.nanmedian(TOIPercent[TOIPercent['Position'] == 'F']['TOI%']) - np.std(TOIPercent[TOIPercent['Position'] == 'D']['TOI%'])
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

player_TOIPerc_F_H, player_TOIPerc_D_H = TOIPercent('EV',names['H'])
player_TOIPerc_F_A, player_TOIPerc_D_A = TOIPercent('EV',names['A'])



## EVEN STRENGTH SIMULATION

# Calculate Proportion of time that each line is on the ice
def propTOIByLine(perTOI,numPlayers):
    TOIByLine = [perTOI[i:i + numPlayers] for i in range(0, len(perTOI), numPlayers)]
    avg_TOIBYLine = [np.mean(x) for x in TOIByLine]
    prop_TOIByLine = [x/(np.sum(avg_TOIBYLine)) for x in avg_TOIBYLine]
    return prop_TOIByLine

propTOIEVH_F = propTOIByLine(player_TOIPerc_F_H,3)
propTOIEVA_F = propTOIByLine(player_TOIPerc_F_A,3)
propTOIEVH_D = propTOIByLine(player_TOIPerc_D_H,2)
propTOIEVA_D = propTOIByLine(player_TOIPerc_D_A,2)


goalCounter = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['H','A'])):
    goalCounter[curSituation[1] + curSituation[0]] = 0


for teami in [['H','A'],['A','H']]:
    if teami[0] == 'H': linepropArray = [propTOIEVH_F,propTOIEVH_D,propTOIEVA_F,propTOIEVA_D]
    else: linepropArray = [propTOIEVA_F,propTOIEVA_D,propTOIEVH_F,propTOIEVH_D]
    totalSituationalSeconds = int(round(EV_TOI_pred*60))
    for dangeri in ['HD','MD','LD']:
        for FLine_O,DPair_O,FLine_D,DPair_D in list(itertools.product([0,1,2,3],[0,1,2],[0,1,2,3],[0,1,2])):
            # Total time on ice for particular line/dpair matchup
            TOIForMatchup = ((((totalSituationalSeconds*linepropArray[0][FLine_O])*(linepropArray[1][DPair_O]))*(linepropArray[2][FLine_D]))*(linepropArray[3][DPair_D]))
            secondsOfMatchup = int(round(TOIForMatchup))
            
            # Get Players on Ice for Each Team
            playersOnIce_O = [names[teami[0]][0:12][i:i + 3] for i in range(0, len(names[teami[0]][0:12]), 3)][FLine_O] + [names[teami[0]][12:18][i:i + 2] for i in range(0, len(names[teami[0]][12:18]), 2)][DPair_O]
            playersOnIce_D = [names[teami[1]][0:12][i:i + 3] for i in range(0, len(names[teami[1]][0:12]), 3)][FLine_D] + [names[teami[1]][12:18][i:i + 2] for i in range(0, len(names[teami[1]][12:18]), 2)][DPair_D]
            
            # Get probability of scoring chance
            # Convert all stats to per second
            def getCurPred(playerStats_relative,x,curStat,homeOrAway):
                try:
                    return playerStats_relative['EV' + homeOrAway].loc[x][curStat + 'adjpermin']
                except:
                    #print(x)
                    return np.nanmedian(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin']) - np.std(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin'])
            
            # Offensive Stat
            CFStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CF',teami[0])/60 for x in playersOnIce_O])
            # Defensive Stat of Opposition
            CAStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CA',teami[1])/60 for x in playersOnIce_D])
            # Average Offensive and Defensive Stats
            ProbSCOccurs = np.mean([CFStat,CAStat])
            
            # Simulate to determine how many scoring chances occur
            numSC = np.sum(np.random.choice([0,1], size=secondsOfMatchup, replace=True, p=[1-ProbSCOccurs,ProbSCOccurs]))
            if numSC > 0:
                # There was at least one scoring chance so calculate probability of scoring on each scoring chance
                def getCurProb(playerStats_relative,x,curStat,homeOrAway):
                    try:
                        return playerStats_relative['EV' + homeOrAway].loc[x][curStat]
                    except:
                        #print(x)
                        return np.nanmedian(playerStats_relative['EV' + homeOrAway][curStat]) - np.std(playerStats_relative['EV' + homeOrAway][curStat])
                        
                
                # Offense Goal For Probability
                GFProbOff = np.mean([getCurProb(playerStats_relative,x,dangeri + '_SC_prob_off',teami[0]) for x in playersOnIce_O])
                # Defense Goal Allowed Probability
                GAProbDef = np.mean([getCurProb(playerStats_relative,x,dangeri + '_SC_prob_def',teami[1]) for x in playersOnIce_D])
                # Average Offensive and Defensive Stats
                ProbGoalOccurs = np.mean([GFProbOff,GAProbDef])
            
                # Simulate whether each scoring chance results in a goal
                numGoals = np.sum(np.random.choice([0,1], size=numSC, replace=True, p=[1-ProbGoalOccurs,ProbGoalOccurs])[0])
                # Populate Array with Results
                goalCounter[teami[0] + dangeri] = goalCounter[teami[0] + dangeri] + numGoals
    













        












'''

goalCounter = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['H','A'])):
    goalCounter[curSituation[1] + curSituation[0]] = 0
    
for i in range(0,int(round(EV_TOI_pred*60))):
    print(i)

    # Given situation, determine who is on the ice for each team
    playersOnIce_H = list(np.random.choice(names['H'][0:12], size=3, replace=False, p=player_TOIPerc_F_H)) + list(np.random.choice(names['H'][12:18], size=2, replace=False, p=player_TOIPerc_D_H))
    playersOnIce_A = list(np.random.choice(names['A'][0:12], size=3, replace=False, p=player_TOIPerc_F_A)) + list(np.random.choice(names['A'][12:18], size=2, replace=False, p=player_TOIPerc_D_A))
    
    
    # Get probability of scoring chance for each team given situational factors
    # Convert all stats to per second
    def getCurPred(playerStats_relative,x,curStat,homeOrAway):
        try:
            return playerStats_relative['EV' + homeOrAway].loc[x][curStat + 'adjpermin']
        except:
            #print(x)
            return np.nanmedian(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin']) - np.std(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin'])
    
    for teami in ['H','A']:
        if teami == 'H': teamOffDef = ['H','A']
        else: teamOffDef = ['A','H']
        
        
        for dangeri in ['HD','MD','LD']:
            # Offensive Stat
            CFStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CF',teamOffDef[0])/60 for x in playersOnIce_H])
            # Defensive Stat of Opposition
            CAStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CA',teamOffDef[1])/60 for x in playersOnIce_A])
            # Average Offensive and Defensive Stats
            ProbSCOccurs = np.mean([CFStat,CAStat])
            
            # Simulate whether a scoring chance happens or not
            DoesSCOccur = np.random.choice([0,1], size=1, replace=True, p=[1-ProbSCOccurs,ProbSCOccurs])[0]
            
            if DoesSCOccur == 1:
                # There was a scoring chance so calculate probability of scoring
                def getCurProb(playerStats_relative,x,curStat,homeOrAway):
                    try:
                        return playerStats_relative['EV' + homeOrAway].loc[x][curStat]
                    except:
                        #print(x)
                        return np.nanmedian(playerStats_relative['EV' + homeOrAway][curStat]) - np.std(playerStats_relative['EV' + homeOrAway][curStat])
                        
                
                # Offense Goal For Probability
                GFProbOff = np.mean([getCurProb(playerStats_relative,x,dangeri + '_SC_prob_off',teamOffDef[0]) for x in playersOnIce_H])
                # Defense Goal Allowed Probability
                GAProbDef = np.mean([getCurProb(playerStats_relative,x,dangeri + '_SC_prob_def',teamOffDef[1]) for x in playersOnIce_A])
                
                # Average Offensive and Defensive Stats
                ProbGoalOccurs = np.mean([GFProbOff,GAProbDef])
            
                # Simulate whether the scoring chance results in a goal
                DoesGoalOccur = np.random.choice([0,1], size=1, replace=True, p=[1-ProbGoalOccurs,ProbGoalOccurs])[0]
                if DoesGoalOccur == 1: 
                    goalCounter[teamOffDef[0] + dangeri] += 1















'''

'''


# Sim whether a high-danger scoring chance occurs
k = 0
counter= 0
for y in range(0,int(round(EV_TOI_pred*60))):
    d = np.random.choice([0,1], size=1, replace=True, p=[1-HDCF_H_Prob_Adj,HDCF_H_Prob_Adj])
    if d == 1:
        k += 1

print(k)


'''




'''


# CALCULATE PREDICTED RELATIVE SCORING CHANCES
print('SC_PRED')
SC_pred_relative = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['CF','CA'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurPred(curSituation,playerStats_relative,x):
        try:
            return playerStats_relative[curSituation[1] + curSituation[2]].loc[x][curSituation[0] + curSituation[3] + 'adjpermin']
        except:
            print(x)
            print(curSituation)
            return np.median(playerStats_relative[curSituation[1] + curSituation[2]][curSituation[0] + curSituation[3] + 'adjpermin']) - np.std(playerStats_relative[curSituation[1] + curSituation[2]][curSituation[0] + curSituation[3] + 'adjpermin'])
    
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurPred(curSituation,playerStats_relative,x) for x in names[curSituation[2]][0:18]]
    # Get weighted scoring chances based on projected mins per player in current situation
    break
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = (np.sum([a * b for a, b in zip(player_TOIPerc, SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))*timeFactor
    


'''






# For every second
# Simulate players on the ice for each team
    







# Simulate whether each team has a corsi for based on CF/CA
# If there is a corsi for, simulate who takes it
# Simulate whether it's a goal based on probability
# Include opposing goalie in probability of scoring % calculation












'''







# CALCULATE PREDICTED RELATIVE SCORING CHANCES
print('SC_PRED')
SC_pred_relative = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['CF','CA'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurPred(curSituation,playerStats_relative,x):
        try:
            return playerStats_relative[curSituation[1] + curSituation[2]].loc[x][curSituation[0] + curSituation[3] + 'adjpermin']
        except:
            print(x)
            print(curSituation)
            return np.median(playerStats_relative[curSituation[1] + curSituation[2]][curSituation[0] + curSituation[3] + 'adjpermin']) - np.std(playerStats_relative[curSituation[1] + curSituation[2]][curSituation[0] + curSituation[3] + 'adjpermin'])
    
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurPred(curSituation,playerStats_relative,x) for x in names[curSituation[2]][0:18]]
    # Get weighted scoring chances based on projected mins per player in current situation
    break
    SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = (np.sum([a * b for a, b in zip(player_TOIPerc, SC_pred_relative[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))*timeFactor
    



SC_pred_compiled_relative = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    if curSituation[1] == 'PP': adjSituation = 'PK'
    elif curSituation[1] == 'PK': adjSituation = 'PP'
    else: adjSituation = 'EV'
    
    SC_pred_compiled_relative[curSituation[0] + curSituation[1] + 'H'] = round((SC_pred_relative[curSituation[0] + curSituation[1] + 'HCF'] + SC_pred_relative[curSituation[0] + adjSituation + 'ACA'])/2)
    SC_pred_compiled_relative[curSituation[0] + curSituation[1] + 'A'] = round((SC_pred_relative[curSituation[0] + curSituation[1] + 'ACF'] + SC_pred_relative[curSituation[0] + adjSituation + 'HCA'])/2)



print('SC_PROB')
SC_prob = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'],['H','A'],['off','def'])):
    if curSituation[1] == 'EV': timeFactor = EV_TOI_pred
    elif (curSituation[1] == 'PP' and curSituation[2] == 'H') or (curSituation[1] == 'PK' and curSituation[2] == 'A'): timeFactor = PP_TOI_pred_H
    elif (curSituation[1] == 'PP' and curSituation[2] == 'A') or (curSituation[1] == 'PK' and curSituation[2] == 'H'): timeFactor = PP_TOI_pred_A
    
    player_TOIPerc = TOIPercent(curSituation[1],timeFactor,names[curSituation[2]])
    def getCurProb(x,OffOrDef):
        try:
            return playerStats_relative[curSituation[1]].loc[x][curSituation[0] + '_SC_prob_' + OffOrDef]
        except:
            return np.median(playerStats_relative[curSituation[1]][curSituation[0] + '_SC_prob_' + OffOrDef]) - np.std(playerStats_relative[curSituation[1]][curSituation[0] + '_SC_prob_' + OffOrDef])
    
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = [getCurProb(x,curSituation[3]) for x in names[curSituation[2]][0:18]]
    SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]] = (np.sum([a * b for a, b in zip(player_TOIPerc, SC_prob[curSituation[0] + curSituation[1] + curSituation[2] + curSituation[3]])])/np.sum(player_TOIPerc))

SC_prob_compiled = dict()
for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
    if curSituation[1] == 'PP': adjSituation = 'PK'
    elif curSituation[1] == 'PK': adjSituation = 'PP'
    else: adjSituation = 'EV'
    SC_prob_compiled[curSituation[0] + curSituation[1] + 'H'] = (SC_prob[curSituation[0] + curSituation[1] + 'Hoff'] + SC_prob[curSituation[0] + adjSituation + 'Adef'])/2
    SC_prob_compiled[curSituation[0] + curSituation[1] + 'A'] = (SC_prob[curSituation[0] + curSituation[1] + 'Aoff'] + SC_prob[curSituation[0] + adjSituation + 'Hdef'])/2







# ADJUST SCORING PROBABILITY BASED ON OPPOSING GOALIE SV%
# Assign Stats of Current Goalies
#goalieStats = assign_stats.assignGoalieStats(goalieStats)

def adjByGoalieStat(curStat,dangeri,goalieStats,SC_prob_compiled,whichGoalie,curSituation):
    try:
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (goalieStats[curSituation].loc[goalieStats[whichGoalie]][dangeri + 'SV%']*SC_prob_compiled[curStat])
    except: # If Goalie has not played then use median of stat
        SC_prob_compiled[curStat] - ( ( np.median( goalieStats[curSituation][dangeri + 'SV%'] ) - np.std( goalieStats[curSituation][dangeri + 'SV%'] ))*SC_prob_compiled[curStat] )
        print('Error: ' + whichGoalie)
    
for curStat in SC_prob_compiled.keys():
    # Subtraction in following equation, because if goalie stat if below average you'd increase scoring probability
    if curStat.endswith('H'):whichGoalie = 'AwayGoalie'
    elif curStat.endswith('A'):whichGoalie = 'HomeGoalie'
    adjByGoalieStat(curStat,curStat[:2],goalieStats,SC_prob_compiled,whichGoalie,curStat[2:4])
            
# ADJUST SCORING PROBABILITY BASED ON REST ADVANTAGE
for curStat in SC_prob_compiled.keys():
    # Adjust based on goal differential of rest advantage
    if curStat.endswith('H'): # Home Team Stat
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] + (SC_prob_compiled[curStat]*restAdj_Goals)
    else:
        SC_prob_compiled[curStat] = SC_prob_compiled[curStat] - (SC_prob_compiled[curStat]*restAdj_Goals)


# SIMULATE OUTCOMES OF EACH SITUATION
print('SIMULATE')
numSims = 20000
compiled_outcomes_H, compiled_outcomes_A = sog_outcome_simulator.simulate_sog(numSims,SC_pred_compiled_relative,SC_prob_compiled)

# Plot Predicted Distribution of Goals for Each Team
assorted_plots.plotPredictedTeamGoals(compiled_outcomes_H,compiled_outcomes_A,homeTeam,awayTeam)


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




# Kelly Criterion Formula
kellyValue_H, kellyValue_A = assorted_minor_functions.kellyCalculation(winProb_H_notie,winProb_A_notie,awayTeam,homeTeam,matchupsInput.loc[curMatchup]['HomeOdds'],matchupsInput.loc[curMatchup]['AwayOdds'])
print('Kelly Values')
print(f"{awayTeam} = {round(kellyValue_A*100,2)}%")
print(f"{homeTeam} = {round(kellyValue_H*100,2)}%")
print()


# Calculated Over/Under Probabilities
overUnderTotal = [x1 + x2 for (x1, x2) in zip(compiled_outcomes_H, compiled_outcomes_A)]
overUnderProb_over = round((len([x for x in overUnderTotal if x > gameOverUnder])/numSims)*100,2)
overUnderProb_under = round((len([x for x in overUnderTotal if x < gameOverUnder])/numSims)*100,2)
overUnderProb_tie = round((len([x for x in overUnderTotal if x == gameOverUnder])/numSims)*100,2)

# Calculated Win Probabilities - Excluding Ties
overUnder_over_notie = round(overUnderProb_over + (overUnderProb_over/(overUnderProb_over + overUnderProb_under))*overUnderProb_tie,2)
overUnder_under_notie = round(overUnderProb_under + (overUnderProb_under/(overUnderProb_over + overUnderProb_under))*overUnderProb_tie,2)
print('When No Tie Possible')
print(f"Over = {overUnder_over_notie}%")
print(f"Under = {overUnder_under_notie}%")
print()

# Kelly Criterion Formula
kellyValue_O, kellyValue_U = assorted_minor_functions.kellyCalculation(overUnder_over_notie,overUnder_under_notie,awayTeam,homeTeam,matchupsInput.loc[curMatchup]['OverOdds'],matchupsInput.loc[curMatchup]['UnderOdds'])
print('Kelly Values')
print(f"Over = {round(kellyValue_O*100,2)}%")
print(f"Under = {round(kellyValue_U*100,2)}%")
print()


#PRINT LINEUPS
assorted_minor_functions.printProjLineups(homeTeam,awayTeam,names)

'''
