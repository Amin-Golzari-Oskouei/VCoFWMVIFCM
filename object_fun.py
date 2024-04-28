def object_fun(row, col, k, Cluster_elem, center_points, data, z, number_viwe, w, beta, alpha2, Cluster_elem_star, pi, q, alpha1, NR, Neig, landa, sample_weight):
    import math
    import numpy as np
    
    j_fun1 = np.zeros([number_viwe])
    j_fun2 = np.zeros([number_viwe])
    j_fun4 = np.zeros([number_viwe])
    j_fun3 = np.zeros([number_viwe]) 
    
    
    for i in range(number_viwe):
        col = data[i].shape[1]
        distance = np.zeros([k, row, col])
        dNK = np.zeros([row, k])
        mf = (Cluster_elem[i] ** 2) + (Cluster_elem_star[i] ** 2)
        
        
        for j in range(k):
            distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
            WBETA = np.transpose(z[i][j, :] ** q)
            WBETA[np.where(np.isinf(WBETA))] = 0
            dNK[:, j] = np.squeeze(np.matmul(np.reshape(distance[j, :, :], (row, col)), np.expand_dims(WBETA, 1))) * sample_weight[i]  
            
            cc = (1-Cluster_elem[i][j,:])**2;
            j_fun4[i] = j_fun4[i] + np.sum(np.transpose(mf[j,:]) * np.sum(cc[Neig[i]],1));
    
        j_fun1[i] = np.sum(np.sum(dNK * np.transpose(mf)))
        
        value = np.transpose(pi[i]) * np.tile((np.exp(1 - ((1/row) * np.sum(pi[i], 1)))), (row, 1))
        j_fun2[i] = (1/row) * np.sum(np.sum(value))
        
       
        for ii in range(number_viwe):
            if ii==i:
                continue
            mf =(((Cluster_elem[i] + Cluster_elem_star[i])-(Cluster_elem[ii] + Cluster_elem_star[ii])) ** 2)
            j_fun3[i] = j_fun3[i] + np.sum(np.sum(dNK * np.transpose(mf)))
               
    return np.sum(w ** beta  * (j_fun1 + (alpha2 * j_fun3) + ((alpha1/NR)*j_fun4) + j_fun2))
