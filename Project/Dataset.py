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
        transforms.ToTensor()  # Convert images to PyTorch tensors
    ])

    return ImageFolder(root='images', transform=transform)


def get_subset_dataset(dataset):
    """
    Create a subset from the dataset and return it
    :param dataset: the dataset to create the subset from
    :return: the subset dataset
    """
    # Create a subset of the dataset
    subset_size = 300  # Use 300 images as a subset
    subset_indices = rnd.sample(range(len(dataset.imgs)), subset_size)  # Randomly sample subset_size indices
    subset_dataset = Subset(dataset, subset_indices)
    return subset_dataset


def get_target_indices(subset_dataset):
    """
    Randomly select 5 images from the subset as target images and return their indices
    :param subset_dataset: the subset dataset
    :return: the indices of the target images in the subset
    """
    # Define the target images
    no_of_images = 5  # Use 5 images as target images
    subset_indices = list(range(len(subset_dataset)))
    target_indices = [subset_indices.pop(rnd.randint(0, len(subset_indices) - 1)) for _ in
                      range(no_of_images)]  # Randomly sample no_of_images indices
    return target_indices
