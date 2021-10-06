#Incorporate high-danger, medium-danger, low-danger scoring chances as that's what you have sv% for for goalies



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

teamStats = dict()
teamStats['EV'] = pd.read_csv('input/2020_2021_TeamStats_Rates_EV.csv').set_index('Team',drop=True)
teamStats['PP'] = pd.read_csv('input/2020_2021_TeamStats_Rates_PP.csv').set_index('Team',drop=True)
teamStats['PK'] = pd.read_csv('input/2020_2021_TeamStats_Rates_PK.csv').set_index('Team',drop=True)

teamStats['EV_cnts'] = pd.read_csv('input/2020_2021_TeamStats_Counts_EV.csv').set_index('Team',drop=True)
teamStats['PP_cnts'] = pd.read_csv('input/2020_2021_TeamStats_Counts_PP.csv').set_index('Team',drop=True)
teamStats['PK_cnts'] = pd.read_csv('input/2020_2021_TeamStats_Counts_PK.csv').set_index('Team',drop=True)


gameResults = pd.read_csv('input/2020_2021_GameResults.csv')

homeTeam = 'Winnipeg Jets'
awayTeam = 'Ottawa Senators'



# CALCULATE PROJECTED SCORING CHANCES AND SCORING PROBABILITY IN EACH SITUATION
def SCNumAndProb(df,stat,curTeam,oppTeam):
    # Projected scoring chances based on average of scoring chances for and those allowed by opponent
    SC_num = (df.loc[curTeam][stat + 'SF/60'] + df.loc[oppTeam][stat + 'SA/60'])/2
    # Scoring probability on scoring chances = goals divided by scoring chances
    SC_prob = ((df.loc[curTeam][stat + 'GF/60'] + df.loc[oppTeam][stat + 'GA/60'])/2)/SC_num
    return SC_num/60, SC_prob

SC_cnts = dict()
SC_prob = dict()

for curStat in ['HD','MD','LD']:
    for curSituation in ['EV','PP','PK']:
        SC_cnts[curStat + curSituation + 'H'], SC_prob[curStat + curSituation + 'H'] = SCNumAndProb(teamStats[curSituation],curStat,homeTeam,awayTeam)
        SC_cnts[curStat + curSituation + 'A'], SC_prob[curStat + curSituation + 'A'] = SCNumAndProb(teamStats[curSituation],curStat,awayTeam,homeTeam)



# MINUTES DISTRIBUTION
# Home Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
total_TOI_H = teamStats['EV_cnts'].loc[homeTeam]['TOI'] + teamStats['PP_cnts'].loc[homeTeam]['TOI'] + teamStats['PK_cnts'].loc[homeTeam]['TOI']
PP_TOI_H = (teamStats['PP_cnts'].loc[homeTeam]['TOI']/total_TOI_H)*60
PK_TOI_H = (teamStats['PK_cnts'].loc[homeTeam]['TOI']/total_TOI_H)*60
# Away Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
total_TOI_A = teamStats['EV_cnts'].loc[awayTeam]['TOI'] + teamStats['PP_cnts'].loc[awayTeam]['TOI'] + teamStats['PK_cnts'].loc[awayTeam]['TOI']
PP_TOI_A = (teamStats['PP_cnts'].loc[awayTeam]['TOI']/total_TOI_A)*60
PK_TOI_A = (teamStats['PK_cnts'].loc[awayTeam]['TOI']/total_TOI_A)*60
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

# SIMULATE OUTCOMES
numSims = 10000
compiled_outcomes_H = []
compiled_outcomes_A = []
for i in range(0,numSims):
    outcomes_H = dict()
    outcomes_A = dict()
    for curStat in SC_cnts.keys():
        if 'A' in curStat:
            outcomes_A[curStat] = np.sum(np.random.choice([0,1], size=int(SC_cnts[curStat]), replace=True, p=[1-SC_prob[curStat], SC_prob[curStat]]))
        else:
            outcomes_H[curStat] = np.sum(np.random.choice([0,1], size=int(SC_cnts[curStat]), replace=True, p=[1-SC_prob[curStat], SC_prob[curStat]]))
    # Append current simresults to list
    compiled_outcomes_H.append(sum(outcomes_H.values()))
    compiled_outcomes_A.append(sum(outcomes_A.values()))

compiled_outcomes_H = [x if x <= 7 else 7 for x in compiled_outcomes_H]
compiled_outcomes_A = [x if x <= 7 else 7 for x in compiled_outcomes_A]

labels, counts = np.unique(compiled_outcomes_H, return_counts=True)
plt.bar(labels-0.2, counts, align='center',width=0.4,alpha=0.5,facecolor='red',edgecolor='black')
plt.gca().set_xticks(labels)
labels, counts = np.unique(compiled_outcomes_A, return_counts=True)
plt.bar(labels+0.2, counts, align='center',width=0.4,alpha=0.5,facecolor='green',edgecolor='black')
colors = {homeTeam:'red', awayTeam:'green'}         
labels = list(colors.keys())
handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
plt.legend(handles, labels)

plt.axvline(np.mean(compiled_outcomes_H), color='red', linewidth=3)
plt.axvline(np.mean(compiled_outcomes_A), color='green', linewidth=3)


plt.show()
    

#print(f"{awayTeam} - {sum(outcomes_A.values())}")
#print(f"{homeTeam} - {sum(outcomes_H.values())}")










