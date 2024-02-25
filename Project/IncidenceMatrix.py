import math

import numpy as np


def calculate_continuous_incidence_matrix(N, k):
    """
    Calculate the continuous incidence matrix H
    :param N: the neighbourhood set matrix N
    :param k: the number of most similar images
    :return: the continuous incidence matrix H
    """
    H = np.zeros((len(N), len(N)))
    for i in range(len(N)):
        for (j, r_i_j) in N[i]:
            H[i][j] = 1 - math.log(r_i_j, k + 1)
    return H
