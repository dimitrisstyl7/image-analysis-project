from DatasetMethods import *
from FeatureVectorMethods import *
if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset()

    # Create a subset of the dataset
    subset_dataset, subset_indices, target_indices = get_subset_dataset(dataset)

    # Extract the feature vectors for all images in the subset dataset
    feature_vectors, target_feature_vectors = extract_feature_vectors(subset_dataset, target_indices)

