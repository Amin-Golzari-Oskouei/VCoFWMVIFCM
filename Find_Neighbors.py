def Find_Neighbors(NR, data, landa, number_viwe):

    from sklearn.metrics import DistanceMetric
    from scipy.spatial.distance import pdist
    import numpy as np
    
    
    result = {}
    dm = {}
    
    def dfun(u, v, landa):
        sqdx = (u-v)**2
        D2 = np.sum(1- np.exp(-1*sqdx*landa),0)
        return D2 
       
    for i in range(number_viwe):
        tmp  = landa[i]
        dist = DistanceMetric.get_metric(dfun, landa=tmp)
        dm[i] = dist.pairwise(data[i], data[i])
        tmp1 = np.argsort(dm[i], 1);
        result[i] = tmp1[:,1:NR+1];
    
    return result, dm