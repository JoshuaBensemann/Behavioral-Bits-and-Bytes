from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


# Define the transformation
def get_transforms(name="train"):
    transforms_list = []

    if name == "train":
        transforms_list.append(transforms.RandomPerspective(p=1))

    elif name == "eval":
        pass

    elif name == "test_1":
        transforms_list.append(v2.Grayscale(3))

    elif name == "test_4":
        pass

    return transforms.Compose(transforms_list + [transforms.ToTensor()])


def get_classes(dataset, verbose=False):
    classes = dataset.classes

    if verbose:
        print("Class Labels:")
        for i, label in enumerate(classes):
            print(f"{i}: {label}")

    return classes


def get_training_dataset(
    dir, transform_set="train", batch_size=1, shuffle=True, verbose=False
):
    transform = get_transforms(transform_set)
    train_dataset = datasets.ImageFolder(dir, transform=transform)
    train_classes = get_classes(train_dataset, verbose=verbose)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataset, train_classes


def get_testing_dataset(
    dir, transform_set="eval", batch_size=1, shuffle=False, verbose=False
):
    transform = get_transforms(transform_set)
    eval_dataset = ImageFolder(dir, transform=transform)
    eval_classes = get_classes(eval_dataset, verbose=verbose)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)

    return eval_dataloader, eval_classes, eval_dataset
