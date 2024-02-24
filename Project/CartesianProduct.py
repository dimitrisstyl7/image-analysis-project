import numpy as np


def calculate_hyperedge_weight(q, H):
    """
    Calculate the weight of a hyperedge
    :param q: hyperedge index
    :param H: continuous incidence matrix
    :return: weight of the hyperedge
    """
    w_e_q = 0
    for i in range(len(H[q])):
        w_e_q += H[q][i]
    return w_e_q


def calculate_pairwise_similarity_relationship(H, q, i, j):
    """
    Calculate the pairwise similarity relationship between two images of a hyperedge
    :param H: continuous incidence matrix
    :param q: hyperedge index
    :param i: first image index
    :param j: second image index
    :return: pairwise similarity relationship
    """
    return calculate_hyperedge_weight(q, H) * H[q][i] * H[q][j]


def calculate_cartesian_product(H):
    """
    Calculate the Cartesian product C
    :param H: continuous incidence matrix
    :return: Cartesian product C
    """
    C = np.zeros((len(H), len(H)))
    for q in range(len(H)):
        for i in range(len(H[q])):
            for j in range(len(H[q])):
                C[i][j] += calculate_pairwise_similarity_relationship(H, q, i, j)
    return C
