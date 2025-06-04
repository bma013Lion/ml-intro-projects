# import pytorch modules for neural network construction
import os
import torch
import torch.nn as nn
import torch.optim as optim
# import pytorch dataloader for data loading
from torch.utils.data import DataLoader
# import torchvision datasets and transformation functions for dataset loading and data preprocessing
import torchvision
# import matplotlib for plotting the results
import matplotlib.pyplot as plt
# import numpy for numerical operations
import numpy as np

# Create output directory
os.makedirs('outputs/pytorch', exist_ok=True)

# set random seed for reproducibility
torch.manual_seed(42)

# check if gpu available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# hyper parameters
# batch size - how many images to pass through the network at once
# smaller batch size can lead to more accurate results, but takes longer to train
batch_size = 64

# learning rate - how quickly the network learns from the data
# if too high, the network may not converge; if too low, it may take too long to converge
learning_rate = 0.001

# number of epochs - how many times to pass through the training set
# more epochs can lead to more accurate results, but takes longer to train
num_epochs = 10

# load dataset FashionMNIST
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,), (0.5,))
])

# load the FashionMNIST dataset
# ---------------------------
# root - the path to the root of the dataset
# train - whether to load the training set or the test set
# download - whether to download the dataset if it is not already downloaded
# transform - the transform to apply to the dataset
train_dataset = torchvision.datasets.FashionMNIST(
    root="./data",  # path to the root of the dataset
    train=True,  # whether to load the training set or the test set
    download=True,  # whether to download the dataset if it is not already downloaded
    transform=transform  # the transform to apply to the dataset
)
  
test_dataset = torchvision.datasets.FashionMNIST(
    root="./data",  # path to the root of the dataset
    train=False,  # whether to load the training set or the test set
    download=True,  # whether to download the dataset if it is not already downloaded
    transform=transform  # the transform to apply to the dataset
)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# image class labels for FashionMNIST
# ---------------------------
# 0: T-shirt/top
# 1: Trouser
# 2: Pullover
# 3: Dress
# 4: Coat
# 5: Sandal
# 6: Shirt
# 7: Sneaker
# 8: Bag
# 9: Ankle boot
image_labels = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# display some training images
def imgshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    
# get some random training images
dataiter = iter(train_loader)
images, labels = next(dataiter)

# show images and print labels
plt.figure(figsize=(10, 5))
for i in range(14):
    plt.subplot(2, 7, i+1)
    imgshow(images[i])
    plt.title(image_labels[labels[i].item()])

plt.tight_layout()
plt.savefig("outputs/pytorch/fashion_mnist_samples_pytorch.png")
plt.close()


