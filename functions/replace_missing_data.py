# REPLACE MISSING DATA WITH MEDIAN

import numpy as np

def replaceMissingValues(curDF):

    curDF = curDF.replace('-',np.nan)
    
    for curCol in [x for x in curDF.columns if x not in ['Team','Position']]:
        curDF[curCol] = curDF[curCol].astype(float)
        curDF[curCol] = curDF.apply(lambda x: np.nanmedian(curDF[curCol]) if ((x['TOI'] < 30) or np.isnan(x[curCol])) else x[curCol], axis=1)
        
    return curDF