import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_data_loaders(batch_size=64, val_split=0.1):
    # Data Augmentation
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # normalization for validation and test
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # load the dataset
    train_dataset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=train_transform
    )
    
    test_dataset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=test_transform
    )
    
    # split the dataset into training and validation sets
    val_size = int(len(train_dataset) * val_split)
    train_size = len(train_dataset) - val_size
    
    train_dataset, val_dataset = random_split(
        train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    
    # class names
    class_names = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')
    
    return train_loader, val_loader, test_loader, class_names
    
    
    x