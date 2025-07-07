import torch
import time
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn_models import (
    Kernel3x3CNN, Kernel5x5CNN, Kernel7x7CNN, MixedKernelCNN,
    ShallowCNN, MediumCNN, DeepCNN, ResNetCNN
)
from utils.comparison_utils import (
    calculate_receptive_field,
    visualize_activations,
    visualize_feature_maps,
    analyze_gradients
)
from utils.training_utils import train_model
from utils.visualization_utils import plot_architecture_results
from pathlib import Path
import matplotlib.pyplot as plt


# Конфигурация
class Config:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 128
        self.learning_rate = 0.001
        self.epochs_kernel = 20
        self.epochs_depth = 30
        self.results_dir = Path("results/architecture_analysis")
        self.plots_dir = Path("plots")
        self.setup_dirs()

    def setup_dirs(self):
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.plots_dir.mkdir(parents=True, exist_ok=True)


def load_cifar10(batch_size=128):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    train_data = datasets.CIFAR10('../data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('../data', train=False, transform=transform)
    return (
        DataLoader(train_data, batch_size=batch_size, shuffle=True),
        DataLoader(test_data, batch_size=batch_size)
    )


def run_kernel_experiment(config):
    train_loader, test_loader = load_cifar10(config.batch_size)

    models = {
        "3x3 Kernels": Kernel3x3CNN(),
        "5x5 Kernels": Kernel5x5CNN(),
        "7x7 Kernels": Kernel7x7CNN(),
        "Mixed Kernels": MixedKernelCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\nTraining {name}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Receptive Field: {calculate_receptive_field(model)}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        train_losses, train_accs, test_accs, _ = train_model(
            model, train_loader, test_loader, config.epochs_kernel,
            optimizer, criterion, config.device
        )
        train_time = time.time() - start_time

        results[name] = {
            'train_accs': train_accs,
            'test_accs': test_accs,
            'train_time': train_time,
            'model': model
        }

        # Визуализация активаций
        sample = next(iter(train_loader))[0][:1].to(config.device)
        fig = visualize_activations(model, sample)
        fig.savefig(config.plots_dir / f"{name.lower().replace(' ', '_')}_activations.png")
        plt.close(fig)

    # Сохранение и визуализация результатов
    plot_architecture_results(results, "Kernel Size Comparison", config)
    torch.save(results, config.results_dir / "kernel_results.pth")


def run_depth_experiment(config):
    train_loader, test_loader = load_cifar10(config.batch_size)

    models = {
        "Shallow (2 conv)": ShallowCNN(),
        "Medium (4 conv)": MediumCNN(),
        "Deep (6 conv)": DeepCNN(),
        "ResNet": ResNetCNN()
    }

    results = {}
    for name, model in models.items():
        print(f"\n=== Training {name} ===")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        criterion = torch.nn.CrossEntropyLoss()

        start_time = time.time()
        train_losses, train_accs, test_accs, grad_flows = train_model(
            model, train_loader, test_loader, config.epochs_depth,
            optimizer, criterion, config.device
        )
        train_time = time.time() - start_time

        results[name] = {
            'train_accs': train_accs,
            'test_accs': test_accs,
            'grad_flows': grad_flows,
            'train_time': train_time,
            'model': model
        }

        # Анализ градиентов
        grad_fig = analyze_gradients(model)
        grad_fig.savefig(config.plots_dir / f"{name.lower().replace(' ', '_')}_gradients.png")
        plt.close(grad_fig)

        # Визуализация feature maps
        sample = next(iter(train_loader))[0][:1].to(config.device)
        fmaps_fig = visualize_feature_maps(model, sample)
        fmaps_fig.savefig(config.plots_dir / f"{name.lower().replace(' ', '_')}_featuremaps.png")
        plt.close(fmaps_fig)

    # Сохранение и визуализация результатов
    plot_architecture_results(results, "Network Depth Comparison", config)
    torch.save(results, config.results_dir / "depth_results.pth")


if __name__ == "__main__":
    config = Config()
    run_kernel_experiment(config)
    # run_depth_experiment(config)