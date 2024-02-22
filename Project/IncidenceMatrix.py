import math

import numpy as np


def calculate_continuous_incidence_matrix(N, k, target_indices):
    """
    Calculate the continuous incidence matrix H
    :param N: the neighbourhood set matrix N
    :param k: the number of most similar images
    :param target_indices: the indices of the target images in the subset dataset
    :return: the continuous incidence matrix H
    """
    H = []
    for i in range(len(N)):
        e = [v for (v, _) in N[i]]
        h = []
        for v in target_indices:
            if v in e:
                j = e.index(v)
                h.append(1 - math.log(N[i][j][1], k + 1))
            else:
                h.append(0)
        H.append(h)
    return np.array(H)
