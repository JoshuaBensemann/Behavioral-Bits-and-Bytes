import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder


# Define the transformation
def get_transforms(name="train"):
    if name == "train":
        transform = transforms.Compose(
            [
                transforms.RandomPerspective(
                    p=1
                ),  # Apply a random perspective transformation
                transforms.ToTensor(),  # Convert the image to a tensor
            ]
        )

    elif name == "test":
        transform = transforms.Compose(
            [
                transforms.ToTensor(),  # Convert the image to a tensor
            ]
        )

    return transform


def get_classes(dataset):
    classes = dataset.classes

    print("Class Labels:")
    for i, label in enumerate(classes):
        print(f"{i}: {label}")

    return classes


def get_training_dataset(dir, transform_set="train", batch_size=1, shuffle=True):
    transform = get_transforms(transform_set)
    dataset = datasets.ImageFolder(dir, transform=transform)
    train_classes = get_classes(dataset)
    train_dataset = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataset, train_classes


def get_testing_dataset(dir, transform_set="test", batch_size=1, shuffle=False):
    transform = get_transforms(transform_set)
    dataset = ImageFolder(dir, transform=transform)
    eval_classes = get_classes(dataset)
    eval_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return eval_dataloader, eval_classes
