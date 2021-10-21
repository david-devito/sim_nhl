
import itertools
import numpy as np


def assignGoalieStats(goalieStats):
    #Mark Home Goalie as Away, and Away goalie as Home to correspond to ooposing skaters
    for curSituation in list(itertools.product(['HD','MD','LD'],['EV','PP','PK'])):
        #Adjust PP/PK for fact that on PK when facing opponent's PP, and vice versa
        adjSituationdict = {'EV':'EV','PP':'PK','PK':'PP'}
        adjSituation = adjSituationdict[curSituation[1]]
        curGoalie = {'A':'HomeGoalie','H':'AwayGoalie'}
        for curGoaliei in curGoalie.keys():
            # Use Median of Stat to fill Goalies who haven't played yet
            try:
                goalieStats[curSituation[0] + curSituation[1] + curGoaliei] = goalieStats[adjSituation].loc[goalieStats[curGoalie[curGoaliei]]][curSituation[0] + 'GSAA/60']
            except:
                goalieStats[curSituation[0] + curSituation[1] + curGoalie] = np.median(goalieStats[adjSituation][curSituation[0] + 'GSAA/60'])
    
    return goalieStats