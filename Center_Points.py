import  numpy as np

def center_Points(data , dm, k, col, row, landa):
    V = []
    a = 2
    delta_prim = np.zeros([row])
    
    # find r
    r = (np.max(dm) - np.amin(np.where(dm == 0, np.inf, dm))) / k

    # calculate P'n'
    p_prim = (dm - r)
    p_prim = np.where(p_prim >= 0, 0, 1)
    p_prim = np.sum(p_prim, axis=1)


    # normalized P'
    p = (p_prim - p_prim.min()) / (p_prim.max() - p_prim.min())  # normalized p_prim

    # delta_prim
    for j in range(row):
        index = np.where(p > p[j])[0]
        if bool(index.all) == True:
            delta_prim[j] = np.max(dm[:, j])
        else:
            delta_prim[j] = np.min(dm[index, j])

    # normalized delta_prim
    delta_prim = (delta_prim - delta_prim.min()) / (delta_prim.max() - delta_prim.min())  # normalized delta_prim

    # t
    t = delta_prim * p

    # dk
    dk = (np.max(dm)) / (a * k)
    
    # dk
    tmp = np.zeros((row, 2))
    tmp [:, 1] = np.linspace(0, row-1, num=row)
    tmp [:, 0] = t
    tmp = tmp[tmp[:, 0].argsort()] [::-1]
    tmp [:, 1]  = np.array(tmp [:, 1], dtype=int)
    
    # sorted t
    t = tmp [:,0]
    
    # sorted data
    data_prim = data [np.array(tmp [:, 1], dtype=int),:]
    
    
    # Find Centers
    V = np.zeros((k, col))
    V [0, :] = data_prim[0, :]
    viewpoint = V [0, :] 
    

    num_cluster = 1
    for i in range (1, row):
        tmp1 = np.zeros(num_cluster)
        for j in range(num_cluster):
            tmp1[j] = np.sum (1-np.exp((-1 * landa * ((data_prim[i,:]-V[j, :]) ** 2))))
                    
        if np.prod(tmp1 > dk):
            V [num_cluster, :] = data_prim[i, :]
            num_cluster += 1
            
        if num_cluster==k:
            break
        
    return V , viewpoint
            