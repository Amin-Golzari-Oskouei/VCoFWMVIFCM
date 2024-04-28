def main(viewpoint, data, center_points, k, t_max, NR, alpha1, landa, q, gama, Neig, number_viwe, row, sample_weight):
    import numpy as np
    import math
    import sys
    from object_fun import object_fun
    
        
#-----------------------------------------------------------------------------------------
    # Other initializations.
    Iter = 1                  # Number of iterations.
    E_w_old = math.inf        # Previous iteration objective (used to check convergence).


    # Weights are uniformly initialized.
    z = {}
    for i in range(number_viwe):
        z[i] = np.ones([k, data[i].shape[1]]) / data[i].shape[1]
        
    
    # Weights are uniformly initialized.
    w = np.ones([number_viwe]) / number_viwe

    # Other initializations.
    Cluster_elem = {}
    Cluster_elem_star = {}
    for i in range(number_viwe): 
        Cluster_elem[i] = np.ones([k, row])/k
        # Update Unk star
        # Cluster_elem_star[i] = 1 - ((1 - (Cluster_elem[i] ** gama)) ** (1/gama))
        Cluster_elem_star[i] = np.ones([k, row])/k
  
        
    dNK = np.zeros([row, k])
    dNK_neig = np.zeros([row, k])
    dw = np.zeros([k])
    part1 = np.zeros([k]) 
    dv = np.zeros([number_viwe])
    pi ={}
    # --------------------------------------------------------------------------
    
    print('Start of Viewpoint-Based Collaborative Feature-Weighted Multi-View Intuitionistic Fuzzy Clustering Using Neighborhood Information iterations')
    print('----------------------------------')

    # the algorithm iteration procedure
    while 1:
        
        alpha2 =  Iter / row
        if number_viwe!=Iter:
            beta = Iter / number_viwe

        # Update the cluster assignments.
        for i in range(number_viwe):    
            col = data[i].shape[1]
            distance = np.zeros([k, row, col])
            
            
            phi = np.zeros([k, row])
            for ii in range(number_viwe):
                if ii==i:
                    continue
                phi = phi + Cluster_elem_star[i] - (Cluster_elem[ii] + Cluster_elem_star[ii]) + 1
                
            phi =  (1 + (phi * alpha2))
            
            # Update the cluster assignments.
            for j in range(k):
                distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
                WBETA = np.transpose(z[i][j, :] ** q)
                WBETA[np.where(np.isinf(WBETA))] = 0
                dNK[:, j] = np.squeeze(np.matmul(np.reshape(distance[j, :, :], (row, col)), np.expand_dims(WBETA, 1)))         
                
                cc = (1-Cluster_elem[i][j,:])**2;
                dNK_neig[:,j]= (dNK[:,j] * (phi[j, :])) + (alpha1/NR) * np.sum(cc[Neig[i]],1);
    
            tmp1 = np.zeros([row, k])
            for j in range(k):
                tmp2 = (dNK_neig / np.transpose(np.tile(dNK_neig[:, j], (k, 1)))) ** (1 / (2 - 1))
                tmp2[np.where(np.isnan(tmp2))] = 0
                tmp2[np.where(np.isinf(tmp2))] = 0
                tmp1 = tmp1 + tmp2
    
            Cluster_elem[i] = np.transpose(1 / tmp1)
            Cluster_elem[i][np.where(np.isnan(Cluster_elem[i]))] = 1
            Cluster_elem[i][np.where(np.isinf(Cluster_elem[i]))] = 1
    
            for j in np.where(dNK_neig == 0)[0]:
                Cluster_elem[i][np.where(dNK_neig[j, :] == 0)[0], j] = 1 / len(np.where(dNK_neig[j, :] == 0)[0])
                Cluster_elem[i][np.where(dNK_neig[j, :] != 0)[0], j] = 0
    
    
            # Update Unk star
            Cluster_elem_star[i] = 1 - ((1 - (Cluster_elem[i] ** gama)) ** (1/gama))
    
            # Update pi
            pi[i] = Cluster_elem_star[i] - Cluster_elem[i]
        
        
        # Calculate the MinMax k-means objective.
        E_w = object_fun(row, col, k, Cluster_elem, center_points, data, z, number_viwe, w, beta, alpha2, Cluster_elem_star, pi, q, alpha1, NR, Neig, landa, sample_weight)

        if math.isnan(E_w) == False:
            print(f'The algorithm objective is E_w={E_w}')

        # Check for convergence. Never converge if in the current (or previous)
        # iteration empty or singleton clusters were detected.
        if (math.isnan(E_w) == False) and (math.isnan(E_w_old) == False) and (abs(1 - E_w / E_w_old) < 10**-6 or Iter >= t_max):

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print(f'Converging for after {Iter} iterations.')
            print(f'The proposed algorithm objective is E_w={E_w}.')
            final = np.zeros([k, row])
            
            for i in range(number_viwe):
                Final = final + (w[i] * Cluster_elem[i])
                           
            return Final

        E_w_old = E_w

        # Update the cluster centers.
        for i in range(number_viwe): 
            
            col = data[i].shape[1]
            tmp5 = np.zeros([k, col])
            tmp6 = np.zeros([k, col])
            
            for ii in range(number_viwe):
                if ii==i:
                    continue
                mf = (((Cluster_elem[i] + Cluster_elem_star[i])-(Cluster_elem[ii] + Cluster_elem_star[ii])) ** 2)
                for j in range(k):
                    tmp5[j,:] = tmp5[j,:]  + ((np.matmul(np.expand_dims(mf[j, :] * sample_weight[i], 0),  (data[i] * (np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))))) + (alpha2 * (tmp5[j,:])))
                    tmp6[j,:] = tmp6[j,:]  + ((np.matmul(np.expand_dims(mf[j, :] * sample_weight[i], 0), ( (np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))))) + (alpha2 * (tmp6[j,:])))  
                
            mf = (Cluster_elem[i] ** 2) + (Cluster_elem_star[i] ** 2)  
            for j in range(k):
                tmp10 = (np.matmul(np.expand_dims(mf[j, :] * sample_weight[i], 0), (data[i] * (np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))))) + (alpha2 * (tmp5[j,:]))
                tmp11 = (np.matmul(np.expand_dims(mf[j, :] * sample_weight[i], 0), ((np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))))) + (alpha2 * (tmp6[j,:]))      
                center_points[i][j, :] = tmp10 / tmp11
            
            
            tmp5 = np.sum((1-np.exp((-1 * np.tile(landa[i], (k, 1))) * ((center_points[i]-np.tile(viewpoint[i], (k, 1))) ** 2))))                 
            center_points[i][np.argmin(tmp5), :] = viewpoint[i]

        # Update the feature weights.
        for i in range(number_viwe):
            col = data[i].shape[1]
            distance = np.zeros([k, row, col])
            dwkm= np.zeros([k, col])

            for ii in range(number_viwe):
                if ii==i:
                    continue
                mf = ((Cluster_elem[i] + Cluster_elem_star[i])-(Cluster_elem[ii] + Cluster_elem_star[ii])) ** 2
                for j in range(k):
                    distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
                    dwkm[j, :] = dwkm[j, :] + np.matmul((mf[j, :] * sample_weight[i]), np.reshape(distance[j, :, :], (row, col)))
                
            mf = (Cluster_elem[i] ** 2) + (Cluster_elem_star[i] ** 2)    
            for j in range(k):
                distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
                dwkm[j, :] = np.matmul((mf[j, :] * sample_weight[i]), np.reshape(distance[j, :, :], (row, col)))  + (alpha2 * dwkm[j, :])
                  
            tmp1 = np.zeros([k, col])
            for j in range(col):
                tmp2 = (dwkm / (np.tile(np.expand_dims(dwkm[:, j],1), (1, col)))) ** (1 / (q - 1))
                tmp2[np.where(np.isnan(tmp2))] = 0
                tmp2[np.where(np.isinf(tmp2))] = 0
                tmp1 = tmp1 + tmp2
    
            z[i] = (1 / tmp1)
            z[i][np.where(np.isnan(z[i]))] = 1
            z[i][np.where(np.isinf(z[i]))] = 1
    
            for j in np.where(dwkm == 0)[0]:
                z[i][j, np.where(dwkm[j, :] == 0)[0]] = 1 / len(np.where(dwkm[j, :] == 0)[0])
                z[i][j, np.where(dwkm[j, :] != 0)[0]] = 0
                
        # check threshold
        for i in range(number_viwe):
            col = data[i].shape[1]
            threshold = 1 / (np.sqrt(row * col))     
            z[i][z[i] < threshold] = 0
            # normalize
            for j in range(k):
                z[i][j, :] = z[i][j, :] / np.sum(z[i][j, :])  
            z[i][np.where(np.isinf(z[i]))] = 1/col
            z[i][np.where(np.isnan(z[i]))] = 1/col
        
        # Update the cluster weights.
        for i in range(number_viwe):
            col = data[i].shape[1]
            distance = np.zeros([k, row, col])
            dw = np.zeros([k])
            
            for ii in range(number_viwe):
                if ii==i:
                    continue
                mf = (((Cluster_elem[i] + Cluster_elem_star[i])-(Cluster_elem[ii] + Cluster_elem_star[ii])) ** 2)
                for j in range(k):
                    distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
                    WBETA = np.transpose(z[i][j, :] ** q)
                    WBETA[np.where(np.isinf(WBETA))] = 0
                    
                    #part 1
                    dw[j] = dw[j] + np.sum(WBETA * np.matmul((mf[j, :] * sample_weight[i]), np.reshape(distance[j, :, :], (row, col))))
            
            mf = (Cluster_elem[i] ** 2) + (Cluster_elem_star[i] ** 2)
            for j in range(k):
                distance[j, :, :] = (1-np.exp((-1 * np.tile(landa[i], (row, 1))) * ((data[i]-np.tile(center_points[i][j, :], (row, 1))) ** 2)))
                WBETA = np.transpose(z[i][j, :] ** q)
                WBETA[np.where(np.isinf(WBETA))] = 0
                
                #part 1
                dw[j] = np.sum(WBETA * np.matmul((mf[j, :] * sample_weight[i]), np.reshape(distance[j, :, :], (row, col))))  + dw[j]
                
                #part 2
                cc = (1-Cluster_elem[i][j,:])**2; 
                part1[j] = (alpha1/NR)*(np.sum(np.transpose(mf[j,:]) * np.sum(cc[Neig[i]],1)));

            value = np.transpose(pi[i]) * np.tile((np.exp(1 - ((1/row) * np.sum(pi[i], 1)))), (row, 1))
            part2 = (1/row) * np.sum(np.sum(value))
            
            dv[i] = np.sum(dw) + np.sum(part1) + part2
                
                
        tmp = np.sum((np.tile(dv, (number_viwe, 1)) / np.transpose(np.tile(dv, (number_viwe, 1)))) ** (1/(beta-1)), axis=0)
        tmp[np.where(np.isnan(tmp))] = 0
        tmp[np.where(np.isinf(tmp))] = 0
        w = 1/tmp
        w[np.where(np.isnan(w))] = 1
        w[np.where(np.isinf(w))] = 1

        if len(np.where(dv == 0)[0]) > 0:
            w[np.where(dv == 0)[0]] = 1 / len(np.where(dv == 0)[0])
            w[np.where(dv != 0)[0]] = 0

        Iter = Iter + 1
