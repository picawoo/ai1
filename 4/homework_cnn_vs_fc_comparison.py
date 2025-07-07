import torch
import time
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from models.fc_models import MNIST_FCN, CIFAR_FCN
from models.cnn_models import MNIST_SimpleCNN, MNIST_ResCNN, CIFAR_ResCNN, CIFAR_RegResCNN
from utils.training_utils import train_model, evaluate
from utils.visualization_utils import plot_results, plot_confusion_matrix
from utils.comparison_utils import calculate_inference_time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
learning_rate = 0.001
epochs = 15


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    dataset = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    return random_split(dataset, [train_size, test_size])


def run_mnist_experiment():
    train_dataset, test_dataset = load_mnist()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    models = {
        "FCN": MNIST_FCN(),
        "SimpleCNN": MNIST_SimpleCNN(),
        "ResCNN": MNIST_ResCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses, train_accs, test_accs, epoch_times = train_model(
            model, train_loader, test_loader, epochs, optimizer, criterion, device
        )

        inference_time = calculate_inference_time(model, test_loader, device)

        results[name] = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "epoch_times": epoch_times,
            "inference_time": inference_time,
            "num_params": sum(p.numel() for p in model.parameters())
        }

    plot_results(results, "MNIST Comparison")


def run_cifar10_experiment():
    train_dataset, test_dataset = load_cifar10()
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    models = {
        "FCN": CIFAR_FCN(),
        "ResCNN": CIFAR_ResCNN(),
        "RegResCNN": CIFAR_RegResCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        train_losses, train_accs, test_accs, epoch_times, grad_flows = train_model(
            model, train_loader, test_loader, epochs, optimizer, criterion, device, track_gradients=True
        )

        results[name] = {
            "train_losses": train_losses,
            "train_accs": train_accs,
            "test_accs": test_accs,
            "epoch_times": epoch_times,
            "grad_flows": grad_flows,
            "model": model
        }

        plot_confusion_matrix(model, test_loader, device)

    plot_results(results, "CIFAR-10 Comparison")


if __name__ == "__main__":
    # run_mnist_experiment()
    run_cifar10_experiment()