import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
import numpy as np


class DepthModel(nn.Module):
    def __init__(self, input_size=2, num_layers=3, hidden_size=128,
                 use_dropout=False, use_batchnorm=False):
        super(DepthModel, self).__init__()
        self.layers = nn.ModuleList()

        self.layers.append(nn.Linear(input_size, hidden_size))
        if use_batchnorm:
            self.layers.append(nn.BatchNorm1d(hidden_size))
        self.layers.append(nn.ReLU())
        if use_dropout:
            self.layers.append(nn.Dropout(0.3))

        # Скрытые слои
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
            if use_batchnorm:
                self.layers.append(nn.BatchNorm1d(hidden_size))
            self.layers.append(nn.ReLU())
            if use_dropout:
                self.layers.append(nn.Dropout(0.3))

        self.layers.append(nn.Linear(hidden_size, 2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def train_model(num_layers, epochs=100, lr=0.01):
    model = DepthModel(num_layers=num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []

    start_time = time.time()

    for epoch in range(epochs):
        # Обучение
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Оценка на тренировочных данных
        train_loss = loss.item()
        train_pred = torch.argmax(outputs, dim=1)
        train_acc = (train_pred == y_train).float().mean()

        # Оценка на тестовых данных
        model.eval()
        with torch.no_grad():
            test_outputs = model(X_test)
            test_loss = criterion(test_outputs, y_test).item()
            test_pred = torch.argmax(test_outputs, dim=1)
            test_acc = (test_pred == y_test).float().mean()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

    training_time = time.time() - start_time

    return {
        'model': model,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'time': training_time
    }


# Генерация данных
X, y = make_moons(n_samples=10000, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train = torch.FloatTensor(X_train)
y_train = torch.LongTensor(y_train)
X_test = torch.FloatTensor(X_test)
y_test = torch.LongTensor(y_test)

# Визуализация данных
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.title("Данные для классификации")
plt.show()
layer_counts = [1, 2, 3, 5, 7]
results = {}

for num_layers in layer_counts:
    print(f"\nTraining model with {num_layers} layers...")
    results[num_layers] = train_model(num_layers)
    final_train_acc = results[num_layers]['train_accs'][-1]
    final_test_acc = results[num_layers]['test_accs'][-1]
    print(f"Model with {num_layers} layers:")
    print(f"Final Training Accuracy: {final_train_acc:.4f}")
    print(f"Final Test Accuracy: {final_test_acc:.4f}")
    print(f"Training Time: {results[num_layers]['time']:.2f} seconds")

plt.figure(figsize=(15, 6))

# График точности на тренировочных данных
plt.subplot(1, 2, 1)
for num_layers in layer_counts:
    plt.plot(results[num_layers]['train_accs'], label=f'{num_layers} layers')
plt.title('Тренировочная точность')
plt.xlabel('Epoch')
plt.ylabel('Точность')
plt.legend()
plt.grid()

# График точности на тестовых данных
plt.subplot(1, 2, 2)
for num_layers in layer_counts:
    plt.plot(results[num_layers]['test_accs'], label=f'{num_layers} layers')
plt.title('Тестовая точность')
plt.xlabel('Epoch')
plt.ylabel('Точность')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()

final_train_accs = [results[n]['train_accs'][-1] for n in layer_counts]
final_test_accs = [results[n]['test_accs'][-1] for n in layer_counts]

plt.figure(figsize=(10, 6))
plt.plot(layer_counts, final_train_accs, 'o-', label='Train Accuracy')
plt.plot(layer_counts, final_test_accs, 'o-', label='Test Accuracy')
plt.title('Точность и число слоев')
plt.xlabel('Число слоев')
plt.ylabel('Точность')
plt.xticks(layer_counts)
plt.legend()
plt.grid()
plt.show()

training_times = [results[n]['time'] for n in layer_counts]

plt.figure(figsize=(10, 6))
plt.plot(layer_counts, training_times, 'o-')
plt.title('Время тренировки и число слоев')
plt.xlabel('Число слоев')
plt.ylabel('Время (сек)')
plt.xticks(layer_counts)
plt.grid()
plt.show()

def retraining_exp():
    datasets = {
        "Moons": make_moons(n_samples=5000, noise=0.3, random_state=42),
        "Circles": make_circles(n_samples=5000, noise=0.2, factor=0.5, random_state=42),
        "Linear": make_classification(n_samples=5000, n_features=2, n_redundant=0,
                                      n_informative=2, random_state=42, n_clusters_per_class=1)
    }

    plt.figure(figsize=(15, 5))
    for i, (name, (X, y)) in enumerate(datasets.items(), 1):
        plt.subplot(1, 3, i)
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
        plt.title(name)
    plt.show()

    def train_and_validate(model, train_loader, test_loader, epochs=100, lr=0.01):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        train_losses, test_losses = [], []
        train_accs, test_accs = [], []

        for epoch in range(epochs):
            model.train()
            train_loss, train_correct = 0.0, 0
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                train_correct += (outputs.argmax(1) == labels).sum().item()

            model.eval()
            test_loss, test_correct = 0.0, 0
            with torch.no_grad():
                for inputs, labels in test_loader:
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item()
                    test_correct += (outputs.argmax(1) == labels).sum().item()

            train_losses.append(train_loss / len(train_loader))
            test_losses.append(test_loss / len(test_loader))
            train_accs.append(train_correct / len(train_loader.dataset))
            test_accs.append(test_correct / len(test_loader.dataset))

        return {
            'train_losses': train_losses,
            'test_losses': test_losses,
            'train_accs': train_accs,
            'test_accs': test_accs
        }

    def run_depth_experiment(dataset_name, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        depths = [1, 2, 3, 5, 7]
        results = {}

        for depth in depths:
            print(f"Training {dataset_name} with depth {depth}...")
            model = DepthModel(num_layers=depth)
            metrics = train_and_validate(model, train_loader, test_loader, epochs=150)
            results[depth] = metrics

        # Визуализация
        plt.figure(figsize=(15, 5))

        # График точности
        plt.subplot(1, 2, 1)
        for depth in depths:
            plt.plot(results[depth]['train_accs'], '--', label=f'Train {depth} layers')
            plt.plot(results[depth]['test_accs'], '-', label=f'Test {depth} layers')
        plt.title(f'{dataset_name} - Точность по глубине')
        plt.xlabel('Epoch')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid()

        # График потерь
        plt.subplot(1, 2, 2)
        for depth in depths:
            plt.plot(results[depth]['train_losses'], '--', label=f'Train {depth} layers')
            plt.plot(results[depth]['test_losses'], '-', label=f'Test {depth} layers')
        plt.title(f'{dataset_name} - Потери по глубине')
        plt.xlabel('Epoch')
        plt.ylabel('Потери')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

        return results

    all_results = {}
    for name, (X, y) in datasets.items():
        all_results[name] = run_depth_experiment(name, X, y)

    def run_regularization_experiment(dataset_name, X, y, depth=5):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Конфигурации моделей
        configs = {
            'Base': {'use_dropout': False, 'use_batchnorm': False},
            'Dropout': {'use_dropout': True, 'use_batchnorm': False},
            'BatchNorm': {'use_dropout': False, 'use_batchnorm': True},
            'Both': {'use_dropout': True, 'use_batchnorm': True}
        }
        results = {}

        for name, config in configs.items():
            print(f"Training {dataset_name} with {name}...")
            model = DepthModel(num_layers=depth, **config)
            metrics = train_and_validate(model, train_loader, test_loader, epochs=150)
            results[name] = metrics

        # Визуализация
        plt.figure(figsize=(15, 5))

        # График точности
        plt.subplot(1, 2, 1)
        for name in configs:
            plt.plot(results[name]['train_accs'], '--', label=f'Train {name}')
            plt.plot(results[name]['test_accs'], '-', label=f'Test {name}')
        plt.title(f'{dataset_name} - Точность с регуляризацией')
        plt.xlabel('Epoch')
        plt.ylabel('Точность')
        plt.legend()
        plt.grid()

        # График потерь
        plt.subplot(1, 2, 2)
        for name in configs:
            plt.plot(results[name]['train_losses'], '--', label=f'Train {name}')
            plt.plot(results[name]['test_losses'], '-', label=f'Test {name}')
        plt.title(f'{dataset_name} - Потери с регуляризацией')
        plt.xlabel('Epoch')
        plt.ylabel('Потери')
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()

        return results

    reg_results = {}
    for name, (X, y) in datasets.items():
        reg_results[name] = run_regularization_experiment(name, X, y)

if __name__ == '__main__':
    retraining_exp()
