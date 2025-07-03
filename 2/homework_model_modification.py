import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
import os

class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer1(x)


def l1_loss(output, target, activations, criterion, l1_lambda):
    main_loss = criterion(output, target)

    if main_loss.dim() > 0:
        main_loss = main_loss.mean()  # Усредняем, если это не скаляр
    l1_penalty = torch.norm(activations, p=1)
    if l1_penalty.dim() > 0:
        l1_penalty = l1_penalty.mean()

    return main_loss + l1_lambda * l1_penalty


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        """
        :param patience: кол-во эпох до остановки без улучшения модели
        :param delta: мин. изменение для фиксирования улучшения
        """
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


class ClassificationMetrics:
    def __init__(self):
        """
        Метрики классификации
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_one_hot(self, labels, num_classes):
        """
        Преобразование в one-hot encoding
        :param labels: тензор меток
        :param num_classes: кол-во классов
        :return: one-hot tensor
        """
        return torch.nn.functional.one_hot(labels, num_classes).float()

    def precision(self, y_true, y_pred, average='macro'):
        """
        Precision метрика
        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param average: тип усреднения
        :return: precision score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)

        num_classes = len(torch.unique(y_true))

        if average == 'micro':
            true_positives = torch.sum(y_true == y_pred).float()
            predicted_positives = len(y_pred)
            return true_positives / predicted_positives if predicted_positives > 0 else 0

        precision_per_class = []
        weights = []

        for c in range(num_classes):
            true_positives = torch.sum((y_true == c) & (y_pred == c)).float()
            predicted_positives = torch.sum(y_pred == c).float()

            precision = true_positives / predicted_positives if predicted_positives > 0 else 0
            precision_per_class.append(precision)

            if average == 'weighted':
                weights.append(torch.sum(y_true == c).float() / len(y_true))

        precision_per_class = torch.tensor(precision_per_class)

        if average == 'macro':
            return torch.mean(precision_per_class).item()
        elif average == 'weighted':
            return torch.sum(precision_per_class * torch.tensor(weights)).item()

        return precision_per_class.tolist()

    def recall(self, y_true, y_pred, average='macro'):
        """
        Recall метрика
        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param average: тип усреднения
        :return: recall score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)

        num_classes = len(torch.unique(y_true))

        if average == 'micro':
            true_positives = torch.sum(y_true == y_pred).float()
            actual_positives = len(y_true)
            return true_positives / actual_positives if actual_positives > 0 else 0

        recall_per_class = []
        weights = []

        for c in range(num_classes):
            true_positives = torch.sum((y_true == c) & (y_pred == c)).float()
            actual_positives = torch.sum(y_true == c).float()

            recall = true_positives / actual_positives if actual_positives > 0 else 0
            recall_per_class.append(recall)

            if average == 'weighted':
                weights.append(torch.sum(y_true == c).float() / len(y_true))

        recall_per_class = torch.tensor(recall_per_class)

        if average == 'macro':
            return torch.mean(recall_per_class).item()
        elif average == 'weighted':
            return torch.sum(recall_per_class * torch.tensor(weights)).item()

        return recall_per_class.tolist()

    def f1_score(self, y_true, y_pred, average='macro'):
        """
        f1 score метрика
        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param average: тип усреднения
        :return: f1 score
        """
        precision = self.precision(y_true, y_pred, average)
        recall = self.recall(y_true, y_pred, average)

        if average in ['macro', 'weighted']:
            return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        elif average == 'micro':
            return precision

        f1_scores = []
        for p, r in zip(precision, recall):
            f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            f1_scores.append(f1)
        return f1_scores

    def roc_auc(self, y_true, y_scores, multi_class='ovr'):
        """
        ROC-AUC метрика
        :param y_true: истинные метки
        :param y_scores: вероятности предсказаний
        :param multi_class: 'ovr' or 'ovo'
        :return: roc-auc score
        """
        y_true = torch.tensor(y_true, dtype=torch.long).cpu().numpy()
        y_scores = torch.tensor(y_scores, dtype=torch.float).cpu().numpy()

        num_classes = y_scores.shape[1]
        y_true_one_hot = self.to_one_hot(torch.tensor(y_true), num_classes).cpu().numpy()

        auc_scores = []

        if multi_class == 'ovr':
            for i in range(num_classes):
                fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_scores[:, i])
                auc_score = auc(fpr, tpr)
                auc_scores.append(auc_score)
            return np.mean(auc_scores)

        elif multi_class == 'ovo':
            auc_sum = 0
            n_pairs = 0
            for i in range(num_classes):
                for j in range(i + 1, num_classes):
                    mask = np.logical_or(y_true == i, y_true == j)
                    y_true_binary = (y_true[mask] == i).astype(int)
                    y_scores_binary = y_scores[mask, i] / (y_scores[mask, i] + y_scores[mask, j])
                    fpr, tpr, _ = roc_curve(y_true_binary, y_scores_binary)
                    auc_sum += auc(fpr, tpr)
                    n_pairs += 1
            return auc_sum / n_pairs if n_pairs > 0 else 0

        return auc_scores

    def confusion_matrix(self, y_true, y_pred, plot=True, save_path=None):
        """
        Визуализация confusion matrix
        :param y_true: истинные метки
        :param y_pred: предсказанные метки
        :param plot: флаг для визуализации
        :param save_path: путь сохраннеия
        :return: confusion matrix
        """
        y_true = torch.tensor(y_true, dtype=torch.long).to(self.device)
        y_pred = torch.tensor(y_pred, dtype=torch.long).to(self.device)

        num_classes = len(torch.unique(y_true))
        cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

        for t, p in zip(y_true, y_pred):
            cm[t, p] += 1

        if plot:
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm.numpy(), annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Confusion matrix сохранена в {save_path}")

            plt.show()
            plt.close()

        return cm