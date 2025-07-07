# homework_custom_layers_experiments.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.custom_layers import CustomConv2d, AttentionBlock, CustomActivation, CustomPooling
from models.residual_blocks import BasicResidualBlock, BottleneckResidualBlock, WideResidualBlock
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 1. Настройки эксперимента
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
lr = 0.001
epochs = 10

# 2. Загрузка данных CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10('./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)


# 3. Тестовые модели
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# 4. Эксперимент с кастомными слоями
def test_custom_layers():
    models = {
    "Standard Conv": SimpleCNN(),
    "Custom Conv": SimpleCNN().conv1 = CustomConv2d(3, 32, 3, padding=1),
    "With Attention": SimpleCNN().conv2 = AttentionBlock(64),
    "Custom Activation": SimpleCNN().relu = CustomActivation(),
    "Custom Pooling": SimpleCNN().maxpool = CustomPooling(2)
    }

    results = {}
    for name, model in models.items():
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        print(f"\nTraining {name}...")
        for epoch in range(epochs):
            model.train()
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # Оценка
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            acc = 100 * correct / total
            results.setdefault(name, []).append(acc)
            print(f"Epoch {epoch + 1}/{epochs}, Acc: {acc:.2f}%")

    # Визуализация
    plt.figure(figsize=(10, 5))
    for name, accs in results.items():
        plt.plot(accs, label=name)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Custom Layers Comparison')
    plt.legend()
    plt.savefig('custom_layers_result.png')
    plt.show()


# 5. Эксперимент с Residual блоками
def test_residual_blocks():
    class ResNet(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.layer1 = block(64, 64)
            self.layer2 = block(64, 128, stride=2)
            self.fc = nn.Linear(128 * 8 * 8, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.layer1(x)
            x = self.layer2(x)
            x = F.avg_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    models = {
        "Basic ResBlock": ResNet(BasicResidualBlock),
        "Bottleneck": ResNet(BottleneckResidualBlock),
        "Wide ResBlock": ResNet(WideResidualBlock)
    }

    # Аналогичный цикл обучения как в test_custom_layers()
    # ...


if __name__ == "__main__":
    test_custom_layers()
    test_residual_blocks()