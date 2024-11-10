import torch
from tqdm import tqdm
import time


# Training function.
def train(model, trainloader, optimizer, criterion, device, scaler):
    model.train()
    print("Training")
    train_running_loss = 0.0
    train_running_correct = 0
    start_time = time.time()
    for i, data in tqdm(enumerate(trainloader), total=len(trainloader)):
        image, labels = data
        image = image.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        # Forward pass with mixed precision.
        with torch.amp.autocast(device_type='mps'):
            outputs = model(image)
            loss = criterion(outputs, labels)
        # Backpropagation and optimization.
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_running_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_running_correct += (preds == labels).sum().item()

    epoch_time = time.time() - start_time
    epoch_loss = train_running_loss / len(trainloader)
    epoch_acc = 100. * (train_running_correct / len(trainloader.dataset))
    print(f"Epoch completed in: {epoch_time:.2f} seconds")
    return epoch_loss, epoch_acc, epoch_time


# Validation function.
def validate(model, testloader, criterion, device):
    model.eval()
    print("Validation")
    valid_running_loss = 0.0
    valid_running_correct = 0

    with torch.no_grad():
        for i, data in tqdm(enumerate(testloader), total=len(testloader)):
            image, labels = data
            image = image.to(device)
            labels = labels.to(device)
            with torch.amp.autocast(device_type='mps'):
                outputs = model(image)
                loss = criterion(outputs, labels)
            valid_running_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            valid_running_correct += (preds == labels).sum().item()

    epoch_loss = valid_running_loss / len(testloader)
    epoch_acc = 100. * (valid_running_correct / len(testloader.dataset))
    return epoch_loss, epoch_acc
