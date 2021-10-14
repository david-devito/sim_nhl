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