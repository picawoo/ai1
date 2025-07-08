import random

import torchvision.transforms.functional as F
from torchvision import transforms

from datasets import CustomImageDataset
from utils import show_many_augs


class RandomBlur:
    """Случайно размывает изображение с заданной вероятностью."""

    def __init__(self, p=0.5, kernel_size=(3, 7), sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        kernel_size = random.choice(range(self.kernel_size[0], self.kernel_size[1] + 1, 2))
        sigma = random.uniform(self.sigma[0], self.sigma[1])

        return F.gaussian_blur(img, kernel_size, sigma)


class RandomContrast:
    """Случайно изменяет контраст изображения."""

    def __init__(self, p=0.5, factor=(0.7, 1.3)):
        self.p = p
        self.factor = factor

    def __call__(self, img):
        if random.random() > self.p:
            return img

        contrast_factor = random.uniform(self.factor[0], self.factor[1])
        return F.adjust_contrast(img, contrast_factor)


class RandomBrightness:
    """Случайно изменяет яркость изображения."""

    def __init__(self, p=0.5, factor=(0.5, 1.5)):
        self.p = p
        self.factor = factor

    def __call__(self, img):
        if random.random() > self.p:
            return img

        brightness_factor = random.uniform(self.factor[0], self.factor[1])
        return F.adjust_brightness(img, brightness_factor)

# Создание аугментаций
augmentations = [
    ("RandomBlur", RandomBlur(p=1.0)),
    ("RandomContrast", RandomContrast(p=1.0)),
    ("RandomBrightness", RandomBrightness(p=1.0))
]

# Применение к изображению
root = './train'
dataset = CustomImageDataset(root, transform=None, target_size=(224, 224))
original_img, label = dataset[0]
augmented_imgs = []
titles = []


for name, aug in augmentations:
    aug_transform = transforms.Compose([
        aug,
        transforms.ToTensor()
    ])
    aug_img = aug_transform(original_img)
    augmented_imgs.append(aug_img)
    titles.append(name)

show_many_augs(original_img, augmented_imgs, titles)