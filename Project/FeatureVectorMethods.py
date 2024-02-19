import torch
import torch.nn as nn
import torchvision.models as models

from torch.utils.data import DataLoader


def extract_feature_vectors(subset_dataset):
    """
    Extract the feature vectors for all images in the subset dataset using the pre-trained ResNet-50 model
    :param subset_dataset: the subset dataset
    :return: the feature vectors for all images in the subset dataset
    """
    # Load the pre-trained model
    model = get_pretrained_model()

    # Create a DataLoader for the subset dataset
    # We set batch_size and num_workers based on our hardware resources
    subset_loader = DataLoader(dataset=subset_dataset, batch_size=64, shuffle=False, num_workers=4)

    # Move the model to the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Compute the feature vectors for all images in the subset dataset using the pre-trained model
    feature_vectors = []

    with torch.no_grad():
        for images, _ in subset_loader:
            # Move the images to the appropriate device
            images = images.to(device)
            # Forward pass through the model to obtain feature vectors
            outputs = model(images)
            # Append the outputs to the feature_vectors list
            feature_vectors.append(outputs)

    # Concatenate the feature vectors into a single tensor
    feature_vectors = torch.cat(feature_vectors, dim=0)

    return feature_vectors


def get_pretrained_model():
    """
    Returns a pre-trained ResNet-50 model with the final layer removed
    :return: the pre-trained model
    """
    # Load the pre-trained model
    model = models.resnet50(weights='ResNet50_Weights.DEFAULT')

    # Replace the final layer with an empty Sequential module, so that we can obtain the feature vectors
    model.fc = nn.Sequential()

    # Set the model to evaluation mode
    model.eval()

    return model
