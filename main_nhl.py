
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


teamStats_EV = pd.read_csv('input/2020_2021_TeamStats_Rates_EV.csv').set_index('Team',drop=True)
teamStats_PP = pd.read_csv('input/2020_2021_TeamStats_Rates_PP.csv').set_index('Team',drop=True)
teamStats_PK = pd.read_csv('input/2020_2021_TeamStats_Rates_PK.csv').set_index('Team',drop=True)

teamStats_EV_cnts = pd.read_csv('input/2020_2021_TeamStats_Counts_EV.csv').set_index('Team',drop=True)
teamStats_PP_cnts = pd.read_csv('input/2020_2021_TeamStats_Counts_PP.csv').set_index('Team',drop=True)
teamStats_PK_cnts = pd.read_csv('input/2020_2021_TeamStats_Counts_PK.csv').set_index('Team',drop=True)


gameResults = pd.read_csv('input/2020_2021_GameResults.csv')

homeTeam = 'Pittsburgh Penguins'
awayTeam = 'Philadelphia Flyers'


# EV STATS
# HOME TEAM SCORING CHANCES
SC_H_EV = (teamStats_EV.loc[homeTeam]['SCSF/60'] + teamStats_EV.loc[awayTeam]['SCSA/60'])/2
SCG_H_EV = (teamStats_EV.loc[homeTeam]['SCGF/60'] + teamStats_EV.loc[awayTeam]['SCGA/60'])/2
SCProb_H_EV = SCG_H_EV/SC_H_EV
SC_A_EV = (teamStats_EV.loc[awayTeam]['SCSF/60'] + teamStats_EV.loc[homeTeam]['SCSA/60'])/2
SCG_A_EV = (teamStats_EV.loc[awayTeam]['SCGF/60'] + teamStats_EV.loc[homeTeam]['SCGA/60'])/2
SCProb_A_EV = SCG_A_EV/SC_A_EV




# PP STATS
# HOME TEAM SCORING CHANCES
SC_H_PP = (teamStats_PP.loc[homeTeam]['SCSF/60'] + teamStats_PK.loc[awayTeam]['SCSA/60'])/2
SCG_H_PP = (teamStats_PP.loc[homeTeam]['SCGF/60'] + teamStats_PK.loc[awayTeam]['SCGA/60'])/2
SCProb_H_PP = SCG_H_PP/SC_H_PP
SC_A_PP = (teamStats_PP.loc[awayTeam]['SCSF/60'] + teamStats_PK.loc[homeTeam]['SCSA/60'])/2
SCG_A_PP = (teamStats_PP.loc[awayTeam]['SCGF/60'] + teamStats_PK.loc[homeTeam]['SCGA/60'])/2
SCProb_A_PP = SCG_A_PP/SC_A_PP

# PK STATS
# HOME TEAM SCORING CHANCES
SC_H_PK = (teamStats_PK.loc[homeTeam]['SCSF/60'] + teamStats_PP.loc[awayTeam]['SCSA/60'])/2
SCG_H_PK = (teamStats_PK.loc[homeTeam]['SCGF/60'] + teamStats_PP.loc[awayTeam]['SCGA/60'])/2
SCProb_H_PK = SCG_H_PK/SC_H_PK
SC_A_PK = (teamStats_PK.loc[awayTeam]['SCSF/60'] + teamStats_PP.loc[homeTeam]['SCSA/60'])/2
SCG_A_PK = (teamStats_PK.loc[awayTeam]['SCGF/60'] + teamStats_PP.loc[homeTeam]['SCGA/60'])/2
SCProb_A_PK = SCG_A_PK/SC_A_PK

# MINUTES DISTRIBUTION
# Home Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
total_TOI_H = teamStats_EV_cnts.loc[homeTeam]['TOI'] + teamStats_PP_cnts.loc[homeTeam]['TOI'] + teamStats_PK_cnts.loc[homeTeam]['TOI']
PP_TOI_avg_H = (teamStats_PP_cnts.loc[homeTeam]['TOI']/total_TOI_H)*60
PK_TOI_avg_H = (teamStats_PK_cnts.loc[homeTeam]['TOI']/total_TOI_H)*60
# Away Situational TOI Ratios - Needs to be ratios as OT causes TOI to go over 60 mins per game
total_TOI_A = teamStats_EV_cnts.loc[awayTeam]['TOI'] + teamStats_PP_cnts.loc[awayTeam]['TOI'] + teamStats_PK_cnts.loc[awayTeam]['TOI']
PP_TOI_avg_A = (teamStats_PP_cnts.loc[awayTeam]['TOI']/total_TOI_A)*60
PK_TOI_avg_A = (teamStats_PK_cnts.loc[awayTeam]['TOI']/total_TOI_A)*60
# Predicted Special Teams TOI
PP_TOI_pred_H = (PP_TOI_avg_H + PK_TOI_avg_A)/2
PP_TOI_pred_A = (PP_TOI_avg_A + PK_TOI_avg_H)/2
EV_TOI_pred = 60 - PP_TOI_pred_H - PP_TOI_pred_A

# ADJUST SCORING CHANCE NUMBERS BASED ON SITUATIONAL MINUTES PREDICTIONS
SC_H_EV_adj = round((SC_H_EV/60)*EV_TOI_pred)
SC_H_PP_adj = round((SC_H_PP/60)*PP_TOI_pred_H)
SC_H_PK_adj = round((SC_H_PK/60)*PP_TOI_pred_A)
SC_A_EV_adj = round((SC_A_EV/60)*EV_TOI_pred)
SC_A_PP_adj = round((SC_A_PP/60)*PP_TOI_pred_A)
SC_A_PK_adj = round((SC_A_PK/60)*PP_TOI_pred_H)

# SIMULATE OUTCOMES OF HOME SCORING CHANCES
SC_H_EV_Outcome = np.sum(np.random.choice([0,1], size=int(SC_H_EV_adj), replace=True, p=[1-SCProb_H_EV, SCProb_H_EV]))
SC_H_PP_Outcome = np.sum(np.random.choice([0,1], size=int(SC_H_PP_adj), replace=True, p=[1-SCProb_H_PP, SCProb_H_PP]))
SC_H_PK_Outcome = np.sum(np.random.choice([0,1], size=int(SC_H_PK_adj), replace=True, p=[1-SCProb_H_PK, SCProb_H_PK]))
SC_A_EV_Outcome = np.sum(np.random.choice([0,1], size=int(SC_A_EV_adj), replace=True, p=[1-SCProb_A_EV, SCProb_A_EV]))
SC_A_PP_Outcome = np.sum(np.random.choice([0,1], size=int(SC_A_PP_adj), replace=True, p=[1-SCProb_A_PP, SCProb_A_PP]))
SC_A_PK_Outcome = np.sum(np.random.choice([0,1], size=int(SC_A_PK_adj), replace=True, p=[1-SCProb_A_PK, SCProb_A_PK]))

print(f"{awayTeam} - {SC_A_EV_Outcome + SC_A_PP_Outcome + SC_A_PK_Outcome}")
print(f"{homeTeam} - {SC_H_EV_Outcome + SC_H_PP_Outcome + SC_H_PK_Outcome}")










