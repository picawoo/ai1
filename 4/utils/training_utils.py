import time
import torch
import numpy as np
from tqdm import tqdm


def train_model(model, train_loader, test_loader, epochs, optimizer, criterion, device, track_gradients=False):
    model.to(device)
    train_losses = []
    train_accs = []
    test_accs = []
    epoch_times = []
    grad_flows = [] if track_gradients else None

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        if track_gradients:
            grad_norms = []

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            if track_gradients:
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                grad_norms.append(total_norm ** 0.5)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_time = time.time() - start_time
        epoch_times.append(epoch_time)

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_acc = evaluate(model, test_loader, device)
        test_accs.append(test_acc)

        if track_gradients:
            grad_flows.append(np.mean(grad_norms))

        print(f"Epoch {epoch + 1}/{epochs}: "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.2f}%, "
              f"Test Acc: {test_acc:.2f}%, "
              f"Time: {epoch_time:.2f}s")

    if track_gradients:
        return train_losses, train_accs, test_accs, epoch_times, grad_flows
    return train_losses, train_accs, test_accs, epoch_times


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100 * correct / total