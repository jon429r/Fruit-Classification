import os
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


def get_data_loaders(batch_size, transform=None,shuffles:list[bool]=[True,False,False]):
    """
    Args:
    - batch_size (int): Batch size for DataLoader.
    - transform The transformations to apply to the images.

    Returns:
    - train_loader, valid_loader, test_loader (DataLoader objects).
    """

    data_dir = "./Fruits_Classification"

    val_transform = transforms.Compose(
        [
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_dir = os.path.join(data_dir, "train")
    valid_dir = os.path.join(data_dir, "valid")
    test_dir = os.path.join(data_dir, "test")

    train_dataset = ImageFolder(root=train_dir, transform=transform)
    valid_dataset = ImageFolder(root=valid_dir, transform=val_transform)
    test_dataset = ImageFolder(root=test_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffles[0])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=shuffles[1])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffles[2])

    return train_loader, valid_loader, test_loader
