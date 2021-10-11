# Simulate Outcomes of Every Shot on Goal

import numpy as np

def simulate_sog(numSims,SC_cnts,SC_prob):
    
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
    
    # Adjust unrealisticly high scores
    compiled_outcomes_H = [x if x <= 7 else 7 for x in compiled_outcomes_H]
    compiled_outcomes_A = [x if x <= 7 else 7 for x in compiled_outcomes_A]
    
    
    return compiled_outcomes_H, compiled_outcomes_A