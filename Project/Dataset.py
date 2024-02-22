import random as rnd

import torchvision.transforms as transforms
from torch.utils.data.dataset import Subset
from torchvision.datasets import ImageFolder


def load_dataset():
    """
    Load the dataset and return it
    :return: the dataset
    """
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224 pixels
        transforms.ToTensor(),  # Convert images to PyTorch tensors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize pixel values
    ])

    return ImageFolder(root='images', transform=transform)


def get_subset_dataset(dataset):
    """
    Create a subset from the dataset and return it
    :param dataset: the dataset to create the subset from
    :return: the subset dataset
    """
    # Create a subset of the dataset
    subset_size = round(len(dataset.imgs) * 0.1)  # Use 10% of the dataset as a subset
    subset_indices = rnd.sample(range(len(dataset.imgs)), subset_size)  # Randomly sample subset_size indices
    subset_dataset = Subset(dataset, subset_indices)

    return subset_dataset


def get_target_indices(subset_dataset):
    """
    Randomly select 5% of the subset as target images and return their indices
    :param subset_dataset: the subset dataset
    :return: the indices of the target images in the subset
    """
    # Define the target images
    no_of_images = round(len(subset_dataset) * 0.05)  # Use 5% of the subset as target images
    target_indices = rnd.sample(range(len(subset_dataset)), no_of_images)  # Randomly sample no_of_images indices

    return target_indices
