import torch
import torch.nn as nn
import torch.optim as optim
from utils import get_data, save_plots
from training_utils import train, validate
from model import TinyVGG_CIFAR10  # Assuming you have the model file created as model.py
from torch.optim.lr_scheduler import StepLR
from torchvision import models
import time

# Check if MPS or CPU is available.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Create model and move to device.
# model = TinyVGG_CIFAR10(num_classes=10).to(device)
model = models.resnet18(weights=None)  # 使用未经预训练的权重
model.fc = nn.Linear(model.fc.in_features, 10)  # 修改输出层
model = model.to(device)

# Get data loaders.
train_loader, valid_loader = get_data(batch_size=32)

# Define the loss function and optimizer.
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scaler = torch.amp.GradScaler()

# Learning rate scheduler.
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

# Train the model and track time for each epoch.
num_epochs = 20
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
epoch_times = []

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")
    train_loss, train_acc, epoch_time = train(model, train_loader, optimizer, criterion, device, scaler)
    valid_loss, valid_acc = validate(model, valid_loader, criterion, device)

    # Append results for plots.
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_acc)
    epoch_times.append(epoch_time)

    print(
        f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}%, Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.2f}%")
    print(f"Time taken for epoch: {epoch_time:.2f} seconds")

    scheduler.step()

# Save the plots for analysis.
save_plots(train_accuracies, valid_accuracies, train_losses, valid_losses, name="resnet18")

# Print total training time.
total_time = sum(epoch_times)
print(f"Total training time: {total_time:.2f} seconds")
print("Training complete")
