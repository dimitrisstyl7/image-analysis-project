import numpy as np


def calculate_similarity_measures(target_feature_vectors, feature_vectors):
    """
    Calculate the pairwise similarity measures between the target images and all other images in the subset dataset
    :param target_feature_vectors: the feature vectors of the target images
    :param feature_vectors: the feature vectors of all images in the subset dataset
    :return: the pairwise similarity measures, format: [ [(i, similarity_measure), ...], ...],
    where i is the index of the image in the subset dataset and similarity_measure is the similarity measure
    between the target image and the image at index i
    """
    # Initialize a numpy array for the pairwise similarity measures
    similarity_measures = []

    # Compute the pairwise similarity measure between each target image and all other images
    for i in range(len(target_feature_vectors)):
        similarity_measures.append([])
        for j in range(len(feature_vectors)):
            similarity_measure = 1 / (np.linalg.norm(target_feature_vectors[i] - feature_vectors[j]) + 1e-8)
            similarity_measures[i].append((j, similarity_measure))

    return similarity_measures
