def parameters(dataset):

    import numpy as np
    import scipy.io
    
    # General Parameters
    t_max = 100                  # maximum number of iterations.
    alpha1 = 2                   # alpha 1 coefficient
    gama = 1

    # specific parameters
    if dataset=='prokaryotic':

        l = 0.1
        q = -1.1
        NR = 2;                      # size of window for finding neighbors
        
        mat = scipy.io.loadmat(f'dataset/{dataset}/{dataset}'+'.mat')
        data = {}
        data[0] = np.array(mat['gene_repert'])
        data[1] = np.array(mat['text'])
        data[2] = np.array(mat['proteome_comp'])
        lable_true = np.squeeze(np.array(mat['truth']))
        
        row = len(lable_true)        # number of samples.
        number_viwe = len(data)      # number of views.
        k = len(set(lable_true))     # number of clusters.

    for i in range(number_viwe):
        data[i] = (data[i]-data[i].min())/(data[i].max()-data[i].min())  # normalized data
    
    
    landa = {}
    for i in range(number_viwe):
        col = data[i].shape[1]
        landa[i] = np.zeros(col)
        landa[i] = l/np.var(data[i], 0)
        landa[i][np.where(np.isinf(landa[i]))] = 1
       
    return k, t_max, NR, alpha1, landa, q, gama, lable_true, number_viwe, data, row