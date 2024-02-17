import numpy as np


def calculate_similarity_measures(target_feature_vectors, feature_vectors):
    """
    Calculate the pairwise similarity measures between the target images and all other images in the subset dataset
    :param target_feature_vectors: the feature vectors of the target images
    :param feature_vectors: the feature vectors of all images in the subset dataset
    :return: the pairwise similarity measures
    """
    # Initialize a numpy array for the pairwise similarity measures
    similarity_measures = np.zeros((len(target_feature_vectors), len(feature_vectors)))

    # Compute the pairwise similarity measure between each target image and all other images
    for i in range(len(target_feature_vectors)):
        for j in range(len(feature_vectors)):
            similarity_measures[i][j] = 1 / (np.linalg.norm(target_feature_vectors[i] - feature_vectors[j]) + 1e-8)

    return similarity_measures
