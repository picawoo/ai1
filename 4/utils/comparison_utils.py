import time
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid


def calculate_inference_time(model, test_loader, device, num_samples=1000):
    model.eval()
    model.to(device)

    dummy_input = torch.randn(1, *next(iter(test_loader))[0].shape[1:]).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    start_time = time.time()
    with torch.no_grad():
        processed = 0
        for images, _ in test_loader:
            images = images.to(device)
            _ = model(images)
            processed += images.size(0)
            if processed >= num_samples:
                break

    total_time = time.time() - start_time
    return total_time / num_samples * 1000


def calculate_receptive_field(model):
    """Вычисляет рецептивное поле модели"""
    rf = 1
    stride = 1
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            k = module.kernel_size[0]
            s = module.stride[0]
            p = module.padding[0]
            rf = rf * s + (k - s)
            stride *= s
    return rf


def visualize_activations(model, sample):
    """Визуализирует активации первого сверточного слоя"""
    activations = []

    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    hook = list(model.children())[0].register_forward_hook(hook_fn)
    with torch.no_grad():
        model(sample)
    hook.remove()

    act = activations[0][0]  # Берем первый образец
    act = act - act.min()
    act = act / act.max()

    fig, ax = plt.subplots(figsize=(12, 6))
    grid = make_grid(act.unsqueeze(1), nrow=8, normalize=False, pad_value=1)
    ax.imshow(grid.permute(1, 2, 0))
    ax.set_title("First Layer Activations")
    ax.axis('off')
    return fig


def analyze_gradients(model):
    """Анализирует распределение градиентов"""
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.abs().mean().item())

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(grads, marker='o')
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean Gradient")
    ax.set_title("Gradient Flow Analysis")
    ax.grid(True)
    return fig


def visualize_feature_maps(model, sample, layer_indices=None, max_maps=32):
    """
    Визуализирует feature maps из разных слоев CNN
    """
    if layer_indices is None:
        # Находим все сверточные слои
        layer_indices = []
        for i, layer in enumerate(model.children()):
            if isinstance(layer, nn.Conv2d):
                layer_indices.append(i)

    activations = {}
    hooks = []

    def hook_fn(layer_idx):
        def hook(module, input, output):
            activations[layer_idx] = output.detach()

        return hook

    for i, layer in enumerate(model.children()):
        if i in layer_indices:
            hooks.append(layer.register_forward_hook(hook_fn(i)))
    with torch.no_grad():
        model(sample.unsqueeze(0))
    for hook in hooks:
        hook.remove()

    # Визуализация
    num_layers = len(activations)
    fig, axes = plt.subplots(num_layers, 1, figsize=(12, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]

    for (layer_idx, acts), ax in zip(activations.items(), axes):
        fmaps = acts[0, :max_maps].cpu()
        fmaps = (fmaps - fmaps.min()) / (fmaps.max() - fmaps.min() + 1e-8)
        grid = make_grid(fmaps.unsqueeze(1), nrow=8, normalize=False, pad_value=1)

        ax.imshow(grid.permute(1, 2, 0))
        ax.set_title(f"Layer {layer_idx} Feature Maps")
        ax.axis('off')

    plt.tight_layout()
    return fig