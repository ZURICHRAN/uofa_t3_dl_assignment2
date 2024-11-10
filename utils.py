import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

plt.style.use("ggplot")

def get_data(batch_size=32):
    # CIFAR10 training dataset.
    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    dataset_train = datasets.CIFAR10(root='data', train=True, download=True, transform=transform)
    dataset_valid = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

    # Create data loaders.
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(dataset_valid, batch_size=batch_size, shuffle=False)
    return train_loader, valid_loader

def save_plots(train_acc, valid_acc, train_loss, valid_loss, name=None):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    # Accuracy plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_acc, color="tab:blue", linestyle="-", label="train accuracy")
    plt.plot(valid_acc, color="tab:red", linestyle="-", label="validation accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(os.path.join("outputs", f"{name}_accuracy.png"))

    # Loss plots.
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="tab:blue", linestyle="-", label="train loss")
    plt.plot(valid_loss, color="tab:red", linestyle="-", label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join("outputs", f"{name}_loss.png"))
