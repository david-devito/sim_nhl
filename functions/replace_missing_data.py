# REPLACE MISSING DATA WITH MEDIAN

import numpy as np

def replaceMissingValues(curDF):

    curDF = curDF.replace('-',np.nan)
    
    for curCol in [x for x in curDF.columns if x not in ['Team','Position','TOI']]:
        curDF[curCol] = curDF[curCol].astype(float)
        curMedian = np.nanmedian(curDF[curDF['TOI'] >= 30][curCol])
        curDF[curCol] = curDF.apply(lambda x: curMedian if ((x['TOI'] < 30) or np.isnan(x[curCol])) else x[curCol], axis=1)
        
    return curDF


def replaceMissingValues_goalies(curDF):

    curDF = curDF.replace('-',np.nan)
    
    for curCol in [x for x in curDF.columns if x not in ['Team','Position','TOI']]:
        curDF[curCol] = curDF[curCol].astype(float)
        curMedian = np.nanmedian(curDF[curDF['TOI'] >= 300][curCol])
        curDF[curCol] = curDF.apply(lambda x: curMedian if ((x['TOI'] < 300) or np.isnan(x[curCol])) else x[curCol], axis=1)
        
    return curDF