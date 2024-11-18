import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder


def get_data_loaders(batch_size, transform=None):
    """
    Args:
    - batch_size (int): Batch size for DataLoader.
    - transform The transformations to apply to the images.

    Returns:
    - train_loader, valid_loader, test_loader (DataLoader objects).
    """

    data_dir = "./Fruits_Classification"

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    valid_dataset = ImageFolder(root=valid_dir, transform=transform)
    test_dataset = ImageFolder(root=test_dir, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader
