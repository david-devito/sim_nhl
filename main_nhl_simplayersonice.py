## SOME OF THE RELATIVE STATS ARE JUST TOO HIGH SO THE PROBABILITY NUMBERS END UP TOO HIGH

## BASELINE STATS NEED TO BE HOME AND AWAY

## split 4 on 4 time out of total Even Strength TOI


# Add paths of additional scripts
import sys
sys.path.append('./functions')

import pandas as pd
import numpy as np
import itertools

import assorted_minor_functions
import load_stats
import time
import assorted_plots

# INPUT
matchupNum = 5

# LOAD STATISTICS FILES
print('LOAD STATS')
teamStats, goalieStats, playerStats_relative, baseline_SC = load_stats.loadStats()

# LOAD MATCHUP INFO
matchupsInput = pd.read_csv('matchups.csv')
curMatchup = matchupsInput.loc[matchupNum]
homeTeam, awayTeam = curMatchup['HomeTeam'], curMatchup['AwayTeam']
goalieStats['HomeGoalie'], goalieStats['AwayGoalie'] = curMatchup['HomeGoalie'], curMatchup['AwayGoalie']
gameOverUnder = curMatchup['OverUnderVal']

# HOME TEAM ADJUSTED WIN AND GOAL PROBABILITIES BASED ON REST ADVANTAGE
restAdj_WP, restAdj_Goals = assorted_minor_functions.restAdvCalc(int(curMatchup['HomeDaysRest']), int(curMatchup['AwayDaysRest']))

## GET LINEUP INFORMATION FROM DFO
names = dict()
names['H'] = assorted_minor_functions.getLineup(homeTeam.replace(' ','-').lower())
names['A'] = assorted_minor_functions.getLineup(awayTeam.replace(' ','-').lower())

# MINUTES DISTRIBUTION BY TEAM
# Predicted Situational TOI
EV_TOI_pred, PP_TOI_pred_H, PP_TOI_pred_A = assorted_minor_functions.TotalTOIBySituation(teamStats,homeTeam,awayTeam)

# Calculate Proportion of even strength time that each line is on the ice
propTOIEVH_F,propTOIEVA_F,propTOIEVH_D,propTOIEVA_D = assorted_minor_functions.get_EV_TOI_dist(names)

# Calculating probability of scoring chance and goal probability with each line matchup on the ice
ProbSCOccurs = dict()
ProbGoalOccurs = dict()
secondsOfMatchup = dict()
for teami in [['H','A'],['A','H']]:
    if teami[0] == 'H': linepropArray = [propTOIEVH_F,propTOIEVH_D,propTOIEVA_F,propTOIEVA_D]
    else: linepropArray = [propTOIEVA_F,propTOIEVA_D,propTOIEVH_F,propTOIEVH_D]
    totalSituationalSeconds = int(round(EV_TOI_pred*60))
    for dangeri in ['HD','MD','LD']:
        for FLine_O,DPair_O,FLine_D,DPair_D in list(itertools.product([0,1,2,3],[0,1,2],[0,1,2,3],[0,1,2])):
            # Total time on ice for particular line/dpair matchup
            TOIForMatchup = ((((totalSituationalSeconds*linepropArray[0][FLine_O])*(linepropArray[1][DPair_O]))*(linepropArray[2][FLine_D]))*(linepropArray[3][DPair_D]))
            secondsOfMatchup[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)] = int(round(TOIForMatchup))
            
            # Get Players on Ice for Each Team
            playersOnIce_O = [names[teami[0]][0:12][i:i + 3] for i in range(0, len(names[teami[0]][0:12]), 3)][FLine_O] + [names[teami[0]][12:18][i:i + 2] for i in range(0, len(names[teami[0]][12:18]), 2)][DPair_O]
            playersOnIce_D = [names[teami[1]][0:12][i:i + 3] for i in range(0, len(names[teami[1]][0:12]), 3)][FLine_D] + [names[teami[1]][12:18][i:i + 2] for i in range(0, len(names[teami[1]][12:18]), 2)][DPair_D]
            
            # Get probability of scoring chance
            # Convert all stats to per second
            def getCurPred(playerStats_relative,x,curStat,homeOrAway):
                try:
                    return playerStats_relative['EV' + homeOrAway].loc[x][curStat + 'adjpermin']
                except:
                    return np.nanmedian(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin']) - np.std(playerStats_relative['EV' + homeOrAway][curStat + 'adjpermin'])
            
            # Offensive Stat
            CFStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CF',teami[0])/60 for x in playersOnIce_O])
            # Defensive Stat of Opposition
            CAStat = np.mean([getCurPred(playerStats_relative,x,dangeri + 'CA',teami[1])/60 for x in playersOnIce_D])
            # Average Offensive and Defensive Stats
            ProbSCOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)] = np.mean([CFStat,CAStat])


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
            ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)] = np.mean([GFProbOff,GAProbDef])
            
            
            #Adjust Probability of Goal based on Opposing Goalie Stat
            def adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs,whichGoalie,curSituation):
                try:
                    return ProbGoalOccurs - (goalieStats[curSituation].loc[goalieStats[whichGoalie]][dangeri + 'SV%']*ProbGoalOccurs)
                except: # If Goalie has not played then use median of stat
                    return ProbGoalOccurs - ( ( np.median( goalieStats[curSituation][dangeri + 'SV%'] ) - np.std( goalieStats[curSituation][dangeri + 'SV%'] ))*ProbGoalOccurs )
                    print('Error: ' + whichGoalie)
            
            if teami[0] == 'H': 
                ProbGoalOccurs_adj = adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)],'AwayGoalie','EV')
                # Adjust Goal Probability based on Rest Adjustment - Add to Home Team Probability
                ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)] = ProbGoalOccurs_adj + (ProbGoalOccurs_adj*restAdj_Goals)
            elif teami[0] == 'A': 
                ProbGoalOccurs_adj = adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)],'HomeGoalie','EV')
                # Adjust Goal Probability based on Rest Adjustment - Subtract from Away Team Probability
                ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)] = ProbGoalOccurs_adj - (ProbGoalOccurs_adj*restAdj_Goals)








compiled_outcomes_H = []
compiled_outcomes_A = []
start = time.time()
numSims = 1
for i in range(0,numSims):
    ## EVEN STRENGTH SIMULATION
    
    # Set up goalCounter to be filled
    goalCounter = dict()
    for curSituation in list(itertools.product(['HD','MD','LD'],['H','A'])):
        goalCounter[curSituation[1] + curSituation[0]] = 0
    
    
    for teami in [['H','A'],['A','H']]:
        if teami[0] == 'H': linepropArray = [propTOIEVH_F,propTOIEVH_D,propTOIEVA_F,propTOIEVA_D]
        else: linepropArray = [propTOIEVA_F,propTOIEVA_D,propTOIEVH_F,propTOIEVH_D]
        totalSituationalSeconds = int(round(EV_TOI_pred*60))
        for dangeri in ['HD','MD','LD']:
            for FLine_O,DPair_O,FLine_D,DPair_D in list(itertools.product([0,1,2,3],[0,1,2],[0,1,2,3],[0,1,2])):
                # Load Probability of Scoring Chance Given Current Matchup
                curProbSC = ProbSCOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)]
                
                # Simulate to determine how many scoring chances occur
                numSC = np.sum(np.random.choice([0,1], size=secondsOfMatchup[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)], replace=True, p=[1-curProbSC,curProbSC]))
                if numSC > 0:
                    curProbGoal = ProbGoalOccurs[teami[0] + dangeri + str(FLine_O) + str(DPair_O) + str(FLine_D) + str(DPair_D)]

                    # Simulate whether each scoring chance results in a goal
                    numGoals = np.sum(np.random.choice([0,1], size=numSC, replace=True, p=[1-curProbGoal,curProbGoal])[0])
                    # Populate Array with Results
                    goalCounter[teami[0] + dangeri] = goalCounter[teami[0] + dangeri] + numGoals
    
    
    # Proportion of PP Time given to Each PP Unit
    PPUnitProportion = [0.70,0.30]
    PKUnitProportion = [0.70,0.30]
    #start = time.time()
    for teami in [['H','A'],['A','H']]:
        # Get PP Units from DFO Scraps
        PPUnits = dict()
        PPUnits[0] = names[teami[0]][18:23]
        PPUnits[1] = names[teami[0]][23:28]
        
        # Determine PK units
        PKTOI_byPlayer = assorted_minor_functions.TOIPercent('PK',names[teami[1]])
        PK_F = [x for i, x in enumerate(names[teami[1]][0:12]) if i in [x[1] for x in sorted( [(x,i) for (i,x) in enumerate(PKTOI_byPlayer[0])], reverse=True )[:4]]]
        PK_D = [x for i, x in enumerate(names[teami[1]][12:18]) if i in [x[1] for x in sorted( [(x,i) for (i,x) in enumerate(PKTOI_byPlayer[1])], reverse=True )[:4]]]
        PKUnits = dict()
        PKUnits[0] = PK_F[0:2] + PK_D[0:2]
        PKUnits[1] = PK_F[2:4] + PK_D[2:4]
        
        
        if teami[0] == 'H': 
            timeFactor = PP_TOI_pred_H
            #print('PP_H')
        else: 
            #print('PP_A')
            timeFactor = PP_TOI_pred_A
        totalSituationalSeconds = int(round(timeFactor*60))
        for dangeri in ['HD','MD','LD']:
            for PPUnit,PKUnit in list(itertools.product([0,1],[0,1])):
                # Total time on ice for particular PPUnit/PKUnit Matchup
                TOIForMatchup = totalSituationalSeconds*PPUnitProportion[PPUnit]*PKUnitProportion[PKUnit]
                secondsOfMatchup_PP = int(round(TOIForMatchup))
                
                # Get Players on Ice for Each Team
                playersOnIce_O = PPUnits[PPUnit]
                playersOnIce_D = PPUnits[PPUnit]
                
                # Get probability of scoring chance
                # Convert all stats to per second
                def getCurPred(playerStats_relative,PPorPK,x,curStat,homeOrAway):
                    try:
                        return playerStats_relative[PPorPK + homeOrAway].loc[x][curStat + 'adjpermin']
                    except:
                        #print(x)
                        return np.nanmedian(playerStats_relative[PPorPK + homeOrAway][curStat + 'adjpermin']) - np.std(playerStats_relative[PPorPK + homeOrAway][curStat + 'adjpermin'])
                
                # Offensive Stat
                CFStat = np.mean([getCurPred(playerStats_relative,'PP',x,dangeri + 'CF',teami[0])/60 for x in playersOnIce_O])
                # Defensive Stat of Opposition
                CAStat = np.mean([getCurPred(playerStats_relative,'PK',x,dangeri + 'CA',teami[1])/60 for x in playersOnIce_D])
                # Average Offensive and Defensive Stats
                ProbSCOccurs_PP = np.mean([CFStat,CAStat])
                
                # Simulate to determine how many scoring chances occur
                numSC = np.sum(np.random.choice([0,1], size=secondsOfMatchup_PP, replace=True, p=[1-ProbSCOccurs_PP,ProbSCOccurs_PP]))
                if numSC > 0:
                    # There was at least one scoring chance so calculate probability of scoring on each scoring chance
                    def getCurProb(playerStats_relative,PPorPK,x,curStat,homeOrAway):
                        try:
                            return playerStats_relative[PPorPK + homeOrAway].loc[x][curStat]
                        except:
                            #print(x)
                            return np.nanmedian(playerStats_relative[PPorPK + homeOrAway][curStat]) - np.std(playerStats_relative[PPorPK + homeOrAway][curStat])
                            
                    
                    # Offense Goal For Probability
                    GFProbOff = np.mean([getCurProb(playerStats_relative,'PP',x,dangeri + '_SC_prob_off',teami[0]) for x in playersOnIce_O])
                    # Defense Goal Allowed Probability
                    GAProbDef = np.mean([getCurProb(playerStats_relative,'PK',x,dangeri + '_SC_prob_def',teami[1]) for x in playersOnIce_D])
                    # Average Offensive and Defensive Stats
                    ProbGoalOccurs_PP = np.mean([GFProbOff,GAProbDef])
                
                    # Adjust Probability of Goal based on Opposing Goalie Stat
                    # Use Opposing goalie SV% on PK, They would be facing a PP
                    def adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs,whichGoalie,curSituation):
                        try:
                            return ProbGoalOccurs - (goalieStats[curSituation].loc[goalieStats[whichGoalie]][dangeri + 'SV%']*ProbGoalOccurs)
                        except: # If Goalie has not played then use median of stat
                            return ProbGoalOccurs - ( ( np.median( goalieStats[curSituation][dangeri + 'SV%'] ) - np.std( goalieStats[curSituation][dangeri + 'SV%'] ))*ProbGoalOccurs )
                            print('Error: ' + whichGoalie)
                    
                    if teami[0] == 'H': 
                        ProbGoalOccurs_PP = adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs_PP,'AwayGoalie','PK')
                        # Adjust Goal Probability based on Rest Adjustment - Add to Home Team Probability
                        ProbGoalOccurs_PP = ProbGoalOccurs_PP + (ProbGoalOccurs_PP*restAdj_Goals)
                    elif teami[0] == 'A': 
                        ProbGoalOccurs_PP = adjByGoalieStat(dangeri,goalieStats,ProbGoalOccurs_PP,'HomeGoalie','PK')
                        # Adjust Goal Probability based on Rest Adjustment - Subtract from Away Team Probability
                        ProbGoalOccurs_PP = ProbGoalOccurs_PP - (ProbGoalOccurs_PP*restAdj_Goals)
                        
                        
                
                    # Simulate whether each scoring chance results in a goal
                    numGoals = np.sum(np.random.choice([0,1], size=numSC, replace=True, p=[1-ProbGoalOccurs_PP,ProbGoalOccurs_PP])[0])
                    # Populate Array with Results
                    goalCounter[teami[0] + dangeri] = goalCounter[teami[0] + dangeri] + numGoals
    
    #print(goalCounter)
    
    
    # Add Goals for Each Team to Compiled Outcomes Arrays
    compiled_outcomes_H.append(goalCounter['HHD'] + goalCounter['HMD'] + goalCounter['HLD'])
    compiled_outcomes_A.append(goalCounter['AHD'] + goalCounter['AMD'] + goalCounter['ALD'])


end = time.time()
print(end-start)

# Plot Predicted Distribution of Goals for Each Team
assorted_plots.plotPredictedTeamGoals(compiled_outcomes_H,compiled_outcomes_A,homeTeam,awayTeam)


# Calculated Win and Tie Probabilities
winProb_H_notie, winProb_A_notie = assorted_minor_functions.calcWinProb(compiled_outcomes_H,compiled_outcomes_A,numSims,awayTeam,homeTeam)


# Kelly Criterion Formula
assorted_minor_functions.kellyCalculation(winProb_H_notie,winProb_A_notie,curMatchup['HomeOdds'],curMatchup['AwayOdds'],homeTeam,awayTeam)


'''
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
kellyValue_O, kellyValue_U = assorted_minor_functions.kellyCalculation(overUnder_over_notie,overUnder_under_notie,awayTeam,homeTeam,curMatchup['OverOdds'],curMatchup['UnderOdds'])
print('Kelly Values')
print(f"Over = {round(kellyValue_O*100,2)}%")
print(f"Under = {round(kellyValue_U*100,2)}%")
print()
'''

#PRINT LINEUPS
assorted_minor_functions.printProjLineups(homeTeam,awayTeam,names)

