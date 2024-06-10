def create_neighborhood_set_matrix(T, k):
    """
    This function creates the neighborhood set matrix N
    :param T: Ranked list of images based on the weights
    :param k: Number of images to be considered in the neighborhood set
    :return: A neighborhood set, format (same as T): [[(image_index, weight), ...], ...]
    """
    return [t[:k] for t in T]
