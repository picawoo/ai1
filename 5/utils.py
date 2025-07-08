import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


def show_images(images, labels=None, nrow=8, title=None, size=128):
    """Визуализация батча изображений."""
    images = images[:nrow]

    resize_transform = transforms.Resize((size, size), antialias=True)
    images_resized = [resize_transform(img) for img in images]
    fig, axes = plt.subplots(1, nrow, figsize=(nrow * 2, 2))
    if nrow == 1:
        axes = [axes]

    for i, img in enumerate(images_resized):
        img_np = img.numpy().transpose(1, 2, 0)
        # Нормализация
        img_np = np.clip(img_np, 0, 1)
        axes[i].imshow(img_np)
        axes[i].axis('off')
        if labels is not None:
            axes[i].set_title(f'Label: {labels[i]}')

    if title:
        fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def show_one_aug(original_img, augmented_img, title="Аугментация"):
    """Визуализирует оригинал и аугментированное изображение."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))

    # Ресайз
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_resized = resize_transform(original_img)
    aug_resized = resize_transform(augmented_img)

    # Оригинал
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    ax1.imshow(orig_np)
    ax1.set_title("Оригинал")
    ax1.axis('off')

    # Аугментированное изображение
    aug_np = aug_resized.numpy().transpose(1, 2, 0)
    aug_np = np.clip(aug_np, 0, 1)
    ax2.imshow(aug_np)
    ax2.set_title(title)
    ax2.axis('off')

    plt.tight_layout()
    plt.show()


def show_many_augs(original_img, augmented_imgs, titles):
    """Визуализация оригинала и нескольких аугментаций."""
    n_augs = len(augmented_imgs)
    fig, axes = plt.subplots(1, n_augs + 1, figsize=((n_augs + 1) * 2, 2))
    to_tensor = transforms.ToTensor()
    resize_transform = transforms.Resize((128, 128), antialias=True)
    orig_tensor = to_tensor(original_img)
    orig_resized = resize_transform(orig_tensor)

    # Преобразуем тензор в numpy для отображения
    orig_np = orig_resized.numpy().transpose(1, 2, 0)
    orig_np = np.clip(orig_np, 0, 1)
    axes[0].imshow(orig_np)
    axes[0].set_title("Оригинал")
    axes[0].axis('off')

    for i, (aug_img, title) in enumerate(zip(augmented_imgs, titles)):
        aug_resized = resize_transform(aug_img)
        aug_np = aug_resized.numpy().transpose(1, 2, 0)
        aug_np = np.clip(aug_np, 0, 1)
        axes[i + 1].imshow(aug_np)
        axes[i + 1].set_title(title)
        axes[i + 1].axis('off')

    plt.tight_layout()
    plt.show()
