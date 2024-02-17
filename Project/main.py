from DatasetMethods import *
if __name__ == '__main__':
    # Load the dataset
    dataset = load_dataset()

    # Create a subset of the dataset
    subset_dataset, subset_indices, target_indices = get_subset_dataset(dataset)

