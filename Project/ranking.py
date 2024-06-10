def get_ranked_list(similarity_measures):
    """
    This function ranks the images based on the similarity measures between the target images and all other images
    :param similarity_measures: Similarity measures between the target images and all other images
    :return: A list of ranked images, format (same as similarity_measures): [[(image_index, similarity_measure), ...], ...]
    """
    T = []
    for row in similarity_measures:
        rank_list = sorted(row, key=lambda x: x[1], reverse=True)
        T.append(rank_list)
    return T


def normalize_ranked_list(T):
    """
    This function normalizes the ranked list T by using the formula: normalized_value = 2 * L - (r_i_j + r_j_i)
    :param T: A list of ranked images, format: [[(image_index, weight), ...], ...]
    :return: A list of normalized ranked images, format: [[(image_index, normalized_weight), ...], ...]
    """
    L = len(T)  # Number of images
    normalized_T = [[] for _ in range(L)]
    for i in range(L):
        for j in range(L):
            r_i_j = T[i][j][1]
            r_j_i = T[j][i][1]
            normalized_value = 2 * L - (r_i_j + r_j_i)
            normalized_T[i].append((T[i][j][0], normalized_value))
    return normalized_T
