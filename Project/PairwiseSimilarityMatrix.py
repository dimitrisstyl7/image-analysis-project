def calculate_pairwise_similarity_matrix(H):
    """
    Calculate the pairwise similarity matrix S
    :param H: continuous incidence matrix
    :return: pairwise similarity matrix S
    """
    H_transpose = H.T  # transpose of H
    S_h = H @ H_transpose  # matrix multiplication H * H_transpose
    S_v = H_transpose @ H  # matrix multiplication H_transpose * H
    S = S_h * S_v  # element-wise multiplication (Hadamard product)
    return S
