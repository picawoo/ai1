import os
from collections import defaultdict

from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(224, 224)):
        """
        Датасет для работы с папками
        :root_dir: директория с классами
        :transform: аугментации
        :target_size: размер ресайза
        """
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size

        self.classes = sorted([d for d in os.listdir(root_dir)
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        # Пути к изображениям
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]

            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_path = os.path.join(class_dir, img_name)
                    self.images.append(img_path)
                    self.labels.append(class_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)

        # Применяем аугментации
        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_names(self):
        """Возвращает список имен классов"""
        return self.classes


def count_images_per_class(dataset_path='train'):
    """Подсчитывает количество изображений в каждом классе датасета."""
    class_counts = defaultdict(int)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Неверная директория")
    classes = [d for d in os.listdir(dataset_path)
               if os.path.isdir(os.path.join(dataset_path, d))]
    for class_name in classes:
        class_dir = os.path.join(dataset_path, class_name)
        # Фильтруем только изображения по расширениям
        image_count = 0
        for file in os.listdir(class_dir):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_count += 1

        class_counts[class_name] = image_count

    return dict(class_counts)


def show_class_nums(class_counts):
    """Визуализирует распределение изображений по классам."""
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    class_names, counts = zip(*sorted_classes)

    plt.figure(figsize=(6, 6))
    plt.subplot(1, 2, 1)
    plt.bar(class_names, counts, color='skyblue')
    plt.title('Кол-во изображений по классам')
    plt.xlabel('Классы')
    plt.ylabel('Кол-во изображений')
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()

count_imgs =count_images_per_class()
count_imgs
show_class_nums(count_imgs)