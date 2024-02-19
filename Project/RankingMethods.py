def get_ranked_list(similarity_measures):
    """
    This function ranks the images based on the similarity measures list
    :param similarity_measures: Similarity measures list
    :return: A list of ranked images, format (same as similarity_measures): [[(image_index, similarity_measure), ...], ...]
    """
    T = []
    for row in similarity_measures:
        rank_list = sorted(row, key=lambda x: x[1], reverse=True)
        T.append(rank_list)
    return T
