from PIL import ImageOps
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2


class Downsample(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size
        self.downsample_size = (112, 112)

    def __call__(self, sample):
        sample.thumbnail(self.downsample_size)
        sample = sample.resize(self.output_size)

        return sample


# Define the transformation
def get_transforms(name="train", verbose=False):
    transforms_list = []

    if name == "train":
        if verbose:
            print("Train")
        transforms_list.append(transforms.RandomPerspective(p=1))

    elif name == "eval":
        if verbose:
            print("Eval")

    elif name == "test_1":
        if verbose:
            print("Test 1")
        transforms_list.append(v2.Grayscale(3))

    elif name == "test_2":
        if verbose:
            print("Test 2")
        transforms_list.append(Downsample((224, 224)))

    elif name == "test_3_vertical_flip":
        if verbose:
            print("Test 3 - Vertical Flip")
        transforms_list.append(transforms.RandomVerticalFlip(p=1))

    elif name == "test_3_horizontal_flip":
        if verbose:
            print("Test 3 - Horizontal Flip")
        transforms_list.append(transforms.RandomHorizontalFlip(p=1))

    elif name == "test_4":
        if verbose:
            print("Test 4")

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
    transform = get_transforms(transform_set, verbose=verbose)
    train_dataset = datasets.ImageFolder(dir, transform=transform)
    train_classes = get_classes(train_dataset, verbose=verbose)
    train_dataset = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_dataset, train_classes


def get_testing_dataset(
    dir, transform_set="eval", batch_size=1, shuffle=False, verbose=False
):
    transform = get_transforms(transform_set, verbose=verbose)
    eval_dataset = ImageFolder(dir, transform=transform)
    eval_classes = get_classes(eval_dataset, verbose=verbose)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=shuffle)

    return eval_dataloader, eval_classes
