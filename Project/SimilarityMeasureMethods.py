import numpy as np


def calculate_similarity_measures(feature_vectors):
    """
    Calculate the pairwise similarity measures for all images in the subset dataset
    :param feature_vectors: the feature vectors of all images in the subset dataset
    :return: the pairwise similarity measures ρ(o_i, o_j), format: [ [(j, similarity_measure), ...], ...],
    where j is the index of the image in the subset dataset and similarity_measure is the pairwise similarity measure
    """
    # Initialize a numpy array for the pairwise similarity measures
    similarity_measures = []

    # Compute the pairwise similarity measure between each target image and all other images
    for i in range(len(feature_vectors)):
        similarity_measures.append([])
        for j in range(len(feature_vectors)):
            similarity_measure = 1 / (np.linalg.norm(feature_vectors[i] - feature_vectors[j]) + 1)
            similarity_measures[i].append((j, similarity_measure))

    return similarity_measures


def normalize_similarity_measures(similarity_measures):
    """
    Normalize the similarity measures
    :param similarity_measures: the pairwise similarity measures ρ(o_i, o_j), format: [ [(j, similarity_measure), ...], ...],
    :return: the normalized similarity measures
    """
    L = len(similarity_measures[0])  # Number of images
    normalized_similarity_measures = [[] for _ in range(L)]
    for i in range(L):
        for j in range(L):
            r_i_j = similarity_measures[i][j][1]
            r_j_i = similarity_measures[j][i][1]
            normalized_value = 2 * L - (r_i_j + r_j_i)
            normalized_similarity_measures[i].append((similarity_measures[i][j][0], normalized_value))
    return normalized_similarity_measures
