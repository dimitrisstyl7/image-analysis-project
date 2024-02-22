def create_neighbourhood_set_matrix(similarity_measures, target_indices, k):
    """
    This function ranks the images based on the similarity measures list and creates the neighbourhood set matrix N
    :param similarity_measures: Similarity measures list
    :param target_indices: Indices of the target images in the subset dataset
    :param k: Number of images to be considered in the neighbourhood set
    :return: A neighbourhood set, format (same as similarity_measures): [[(image_index, similarity_measure), ...], ...]
    """
    temp_list = []
    for row in similarity_measures:
        ranked_list = sorted(row, key=lambda x: x[1])  # rank the images based on the similarity measures
        temp_list.append(ranked_list[:k])  # Take the top k images
    return [temp_list[i] for i in target_indices]  # Create the neighbourhood set matrix N
