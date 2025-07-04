import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import time
import numpy as np

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)


class WidthVariableNet(nn.Module):
    def __init__(self, layer_sizes):
        super(WidthVariableNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, layer_sizes[0])
        self.relu1 = nn.ReLU()

        self.fc2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.relu2 = nn.ReLU()

        self.fc3 = nn.Linear(layer_sizes[1], layer_sizes[2])
        self.relu3 = nn.ReLU()

        self.fc_out = nn.Linear(layer_sizes[2], 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.fc_out(x)
        return x


def train_and_evaluate(layer_sizes, epochs=10):
    model = WidthVariableNet(layer_sizes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel with layer sizes {layer_sizes}")
    print(f"Total parameters: {total_params:,}")

    train_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        model.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

        test_acc = test_correct / test_total
        test_accs.append(test_acc)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")

    training_time = time.time() - start_time
    print(f"Training time: {training_time:.2f} seconds")

    return {
        'layer_sizes': layer_sizes,
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'time': training_time,
        'params': total_params
    }

width_configs = {
    'Narrow': [64, 32, 16],
    'Medium': [256, 128, 64],
    'Wide': [1024, 512, 256],
    'Very Wide': [2048, 1024, 512]
}

results = {}

for name, sizes in width_configs.items():
    print(f"\n=== Training {name} network: {sizes} ===")
    results[name] = train_and_evaluate(sizes)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name in width_configs:
    plt.plot(results[name]['train_accs'], label=f'{name} ({results[name]["params"]/1e6:.1f}M params)')
plt.title('Training Accuracy by Network Width')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
for name in width_configs:
    plt.plot(results[name]['test_accs'], label=f'{name} ({results[name]["params"]/1e6:.1f}M params)')
plt.title('Test Accuracy by Network Width')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

names = list(width_configs.keys())
final_test_accs = [results[name]['test_accs'][-1] for name in names]
times = [results[name]['time'] for name in names]
params = [results[name]['params'] for name in names]

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
bars = plt.bar(names, final_test_accs)
plt.title('Final Test Accuracy')
plt.ylabel('Accuracy')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.4f}', ha='center', va='bottom')
plt.grid(axis='y')

plt.subplot(1, 2, 2)
bars = plt.bar(names, times)
plt.title('Training Time')
plt.ylabel('Seconds')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s', ha='center', va='bottom')
plt.grid(axis='y')

plt.tight_layout()
plt.show()

print("\nModel Comparison:")
print("{:<10} {:<15} {:<15} {:<15}".format('Width', 'Parameters', 'Test Acc', 'Time (s)'))
for name in names:
    print("{:<10} {:<15,} {:<15.4f} {:<15.2f}".format(
        name, results[name]['params'], results[name]['test_accs'][-1], results[name]['time']))