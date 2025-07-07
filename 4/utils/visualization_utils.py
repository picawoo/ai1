import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from sklearn.metrics import confusion_matrix


def plot_results(results, title):
    plt.figure(figsize=(15, 10))

    # Точность
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_accs'], label=f'{name} (train)')
        if 'test_accs' in res:
            plt.plot(res['test_accs'], '--', label=f'{name} (test)')
    plt.title('Точность')
    plt.xlabel('Эпохи')
    plt.ylabel('Точность (%)')
    plt.legend()

    # Потери
    plt.subplot(2, 2, 2)
    for name, res in results.items():
        plt.plot(res['train_losses'], label=name)
    plt.title('Потери при обучении')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()

    # Градиенты
    if 'grad_flows' in next(iter(results.values())):
        plt.subplot(2, 2, 3)
        for name, res in results.items():
            plt.plot(res['grad_flows'], label=name)
        plt.title('L2 norm')
        plt.xlabel('Эпохи')
        plt.ylabel('Градиент')
        plt.legend()

    # Общая таблица
    plt.subplot(2, 2, 4)
    summary_data = []
    for name, res in results.items():
        row = [name]
        if 'num_params' in res:
            row.append(f"{res['num_params']:,}")
        else:
            row.append("-")

        if 'train_accs' in res:
            row.append(f"{res['train_accs'][-1]:.2f}%")
        else:
            row.append("-")

        if 'test_accs' in res:
            row.append(f"{res['test_accs'][-1]:.2f}%")
        else:
            row.append("-")

        if 'epoch_times' in res:
            row.append(f"{sum(res['epoch_times']):.2f}s")
        else:
            row.append("-")

        if 'inference_time' in res:
            row.append(f"{res['inference_time']:.2f}ms")
        elif 'grad_flows' in res:
            row.append(f"{res['grad_flows'][-1]:.2f}")
        else:
            row.append("-")

        summary_data.append(row)

    columns = ['Model', 'Params', 'Train Acc', 'Test Acc', 'Total Time', 'Inference/Grad']
    plt.table(cellText=summary_data, colLabels=columns, loc='center')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def plot_architecture_results(results, title, config):
    """Визуализирует результаты сравнения архитектур"""
    plt.figure(figsize=(18, 12))

    # График точности
    plt.subplot(2, 2, 1)
    for name, res in results.items():
        plt.plot(res['train_accs'], label=f'{name} (train)')
        if 'test_accs' in res:
            plt.plot(res['test_accs'], '--', label=f'{name} (test)')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    # График времени обучения
    plt.subplot(2, 2, 2)
    times = [res['train_time'] for res in results.values()]
    plt.bar(results.keys(), times)
    plt.title('Training Time')
    plt.ylabel('Time (s)')

    # График градиентов (если есть)
    if 'grad_flows' in next(iter(results.values())):
        plt.subplot(2, 2, 3)
        for name, res in results.items():
            plt.plot(res['grad_flows'], label=name)
        plt.title('Gradient Flow')
        plt.xlabel('Epoch')
        plt.ylabel('Gradient Norm')
        plt.legend()

    # Сводная таблица
    plt.subplot(2, 2, 4)
    summary_data = []
    for name, res in results.items():
        row = [name]
        row.append(f"{res['train_accs'][-1]:.2f}%")
        if 'test_accs' in res:
            row.append(f"{res['test_accs'][-1]:.2f}%")
        else:
            row.append("N/A")
        row.append(f"{res['train_time']:.2f}s")
        if 'grad_flows' in res:
            row.append(f"{res['grad_flows'][-1]:.4f}")
        summary_data.append(row)

    columns = ['Model', 'Train Acc', 'Test Acc', 'Train Time', 'Grad Norm']
    plt.table(cellText=summary_data, colLabels=columns, loc='center')
    plt.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(config.plots_dir / f"{title.lower().replace(' ', '_')}.png")
    plt.close()