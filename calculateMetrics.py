def calculateMetrics(lable_pre, lable_true, row):
    import numpy as np
    import itertools
    from sklearn.metrics import jaccard_score, fowlkes_mallows_score, accuracy_score, precision_score, recall_score, f1_score, rand_score, adjusted_rand_score
    from sklearn.metrics.cluster import v_measure_score, davies_bouldin_score, silhouette_score

    cluster_names = np.unique(lable_pre)
    ACC_score = 0
    NMI_score = 0
    PRE_score = 0
    REC_score = 0
    F_score = 0
    DBS_score= 0
    SI_score = 0

    [a] = np.unique(lable_true).shape
    permut = itertools.permutations(np.unique(lable_true))
    perm = np.empty((0, a))
    for p in permut:
        perm = np.append(perm, np.atleast_2d(p), axis=0)

    [pN, pM] = perm.shape

    for i in range(pN):
        flipped_labels = np.zeros([row])
        if cluster_names.size > 1:
            for cl in range(pM):
                flipped_labels[lable_pre == cluster_names[cl]] = perm[i, cl]
        else:
            flipped_labels[1: row] = cluster_names

        testAcc = accuracy_score(lable_true, flipped_labels)
        testNMI = v_measure_score(lable_true, flipped_labels)
        testpre = precision_score(lable_true, flipped_labels, average='macro')
        testrec = recall_score(lable_true, flipped_labels, average='macro')
        testf1s = f1_score(lable_true, flipped_labels, average='macro')
        testrs  = rand_score(lable_true, flipped_labels)
        testARI = adjusted_rand_score(lable_true, flipped_labels)
        testFMI = fowlkes_mallows_score(lable_true, flipped_labels)
        testJI = jaccard_score(lable_true, flipped_labels, average='macro')


        if testf1s > F_score:
            NMI_score = testNMI
            ACC_score = testAcc
            PRE_score = testpre
            REC_score = testrec
            F_score = testf1s
            R_index = testrs
            AR_index = testARI
            FMI = testFMI
            JI = testJI

    ans = np.array([ACC_score, NMI_score, PRE_score, REC_score, F_score, R_index, AR_index, FMI, JI])
    return ans
