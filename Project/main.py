import matplotlib.pyplot as plt

from CartesianProduct import *
from Dataset import *
from FeatureVector import *
from IncidenceMatrix import *
from NeighborhoodSet import *
from PairwiseSimilarityMatrix import *
from Ranking import *
from SimilarityMeasure import *


def LHRR(T, iterations):
    """
    This function implements the Log-based Hypergraph of Ranking References (LHRR) algorithm
    :param T: A list of ranked images, format: [[(image_index, weight), ...], ...]
    :param iterations: Number of iterations to perform
    :return: A list of ranked images, format: [[(image_index, weight), ...], ...] and the number of
    similar images considered in the neighborhood set
    """
    for iteration in range(iterations):
        print("\nStarting iteration " + str(iteration + 1) + "/" + str(iterations) + ":")

        # Rank normalization
        print("\tNormalizing the ranked list T...")
        T = normalize_ranked_list(T)

        # Create the neighborhood set matrix N
        print("\tCreating the neighborhood set matrix N...")
        k = 5  # k most similar images to consider in the neighborhood set
        N = create_neighborhood_set_matrix(T, k)

        # Calculate continuous incidence matrix H
        print("\tCalculating the continuous incidence matrix H...")
        H = calculate_continuous_incidence_matrix(N, k)

        # Calculate the pairwise similarity matrix S
        print("\tCalculating the pairwise similarity matrix S...")
        S = calculate_pairwise_similarity_matrix(H)

        # Calculate the Cartesian product C
        print("\tCalculating the Cartesian product C...")
        C = calculate_cartesian_product(H)

        # Calculate the affinity matrix W
        print("\tCalculating the affinity matrix W...")
        W = C * S

        # Update the ranked list T with the new weights
        print("\tUpdating the ranked list T with the new weights...")
        for i in range(len(W)):
            for j in range(len(W[i])):
                T[i][j] = (T[i][j][0], W[i][j])

        # Sort the ranked list T
        print("\tSorting the ranked list T...")
        T = [sorted(t, key=lambda x: x[1], reverse=True) for t in T]
    return T, k


def assign_weights_with_relevance(no_of_images, relevance_scores):
    """
    Assign weights to the images based on their index and relevance scores
    :param no_of_images: Number of images
    :param relevance_scores: List of relevance scores
    :return: List of weights
    """
    return [relevance_scores[i] * (no_of_images - i) for i in range(no_of_images)]


def show_images(target_image, subset_dataset, categories):
    _, axs = plt.subplots(1, len(target_image), figsize=(14, 4))
    plt.gcf().canvas.manager.set_window_title(f"Target image {target_image[0][0]}")

    # Target image category
    target_category_idx = subset_dataset.dataset.samples[subset_dataset.indices[target_image[0][0]]][1]

    # Initialize the relevance scores
    relevance_scores = []

    # Iterate through the target images
    for ax, (img_idx, score) in zip(axs, target_image):
        # Retrieve the image from the dataset
        image, _ = subset_dataset.dataset[subset_dataset.indices[img_idx]]

        # Convert the image tensor to numpy array and transpose it
        image = np.transpose(image.numpy(), (1, 2, 0))

        # Plot the image
        ax.imshow(image)

        # Set the title of the image
        category_idx = subset_dataset.dataset.samples[subset_dataset.indices[img_idx]][1]
        category = categories[category_idx]
        title = f'Category: {category}'
        if img_idx == target_image[0][0]:
            title += ' (target image)'
        ax.set_title(title)

        # Remove the axis
        ax.axis('off')

        # Calculate the relevance score
        score = 1 if category_idx == target_category_idx else 0
        relevance_scores.append(score)

    # Assign weights to the images based on their index and relevance scores
    weights = assign_weights_with_relevance(len(target_image), relevance_scores)

    # Get the accuracy of the algorithm based on the weights
    accuracy = sum(weights) / sum(range(1, len(target_image) + 1))

    # Show the accuracy on the plot
    plt.suptitle(f'Accuracy: {accuracy:.2f}')

    # Show the images
    plt.show()


def get_target_images(T, target_indices, k):
    """
    Get the target images from the ranked list
    :param T: A list of ranked images, format: [[(image_index, weight), ...], ...]
    :param target_indices: The indices of the target images in the subset
    :param k: The number of similar images considered in the neighborhood set
    :return: The target images with the k most similar images
    """
    return [T[idx][:k] for idx in target_indices]


if __name__ == '__main__':
    # Load the dataset
    print("\nFetching the dataset...")
    dataset = load_dataset()

    # Create a subset of the dataset
    print("Creating a subset of the dataset...")
    subset_dataset = get_subset_dataset(dataset)

    # Get the indices of the target images in the subset dataset
    print("Selecting the target images...")
    target_indices = get_target_indices(subset_dataset)

    # Extract the feature vectors for all images in the subset dataset
    print("Extracting the feature vectors...")
    feature_vectors = extract_feature_vectors(subset_dataset)

    # Calculate the similarity measures for all images
    print("Calculating the similarity measures...")
    similarity_measures = calculate_similarity_measures(feature_vectors)

    # Rank the images based on the similarity measures
    print("Ranking the images based on the similarity measures...")
    T = get_ranked_list(similarity_measures)

    # Iterate through the main steps of the algorithm
    T, k = LHRR(T, iterations=5)

    # Get the target images from the ranked list
    target_images = get_target_images(T, target_indices, k)

    # Get all categories names from the dataset
    categories = subset_dataset.dataset.classes

    # Show the target images and the k most similar images
    for target_image in target_images:
        show_images(target_image, subset_dataset, categories)
