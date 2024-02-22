from DatasetMethods import *
from FeatureVectorMethods import *
from IncidenceMatrixMethods import *
from NeighbourhoodSetMethods import *
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
    normalized_similarity_measures = normalize_similarity_measures(similarity_measures)

    # Create the neighbourhood set matrix N (inside this method, we rank the images based on the similarity measures)
    k = len(target_indices)  # k most similar images
    N = create_neighbourhood_set_matrix(normalized_similarity_measures, target_indices, k)

    # Calculate continuous incidence matrix H
    H = calculate_continuous_incidence_matrix(N, k, target_indices)
