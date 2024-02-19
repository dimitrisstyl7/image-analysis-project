from DatasetMethods import *
from FeatureVectorMethods import *
from RankingMethods import *
from SimilarityMeasureMethods import *

if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset()

    # Create a subset of the dataset
    subset_dataset = get_subset_dataset(dataset)

    # Get the indices of the target images in the subset dataset
    target_indices = get_target_indices(subset_dataset)

    # Extract the feature vectors for all images in the subset dataset
    feature_vectors = extract_feature_vectors(subset_dataset)

    # Calculate the similarity measures for all images
    similarity_measures = calculate_similarity_measures(feature_vectors)

    # Normalize the similarity measures
    normalize_similarity_measures(similarity_measures)

    # Rank the images based on the similarity measures
    T = get_ranked_list(similarity_measures)
