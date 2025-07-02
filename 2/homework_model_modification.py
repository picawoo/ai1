import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import make_regression_data, mse, log_epoch, RegressionDataset, make_classification_data, \
    ClassificationDataset
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class LinearRegression(nn.Module):
    def __init__(self, in_features, l1_lambda=0.01, l2_lambda=0.01):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

    def forward(self, x):
        return self.linear(x)

    def regularization(self):
        l1_loss = sum(p.abs().sum() for p in self.parameters())
        l2_loss = sum(p.pow(2).sum() for p in self.parameters())
        return self.l1_lambda * l1_loss + self.l2_lambda * l2_loss


def linear_reg():
    """
    Расширенная версия линейной регрессии
    """
    # Генерируем данные
    X, y = make_regression_data(n=200)

    # Создаём датасет и даталоадер
    dataset = RegressionDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LinearRegression(in_features=1, l1_lambda=0.1, l2_lambda=0.1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Параметры ранней остановки
    patience = 5
    best_loss = np.inf
    min_delta = 0.001
    patience_cnt = 0

    # Обучаем модель
    epochs = 100
    for epoch in range(1, epochs + 1):
        total_loss = 0

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model(batch_X)
            mse_loss = criterion(y_pred, batch_y)
            reg_loss = model.regularization()
            loss = mse_loss + reg_loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / (i + 1)

        # Проверяем улучшилась ли модель
        if avg_loss + min_delta < best_loss:
            best_loss = avg_loss
            patience_cnt = 0
            torch.save(model.state_dict(), 'top_model.pth')
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f'Остановка на {epoch}')
                break

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss)

    # Загружаем лучшие веса
    model.load_state_dict(torch.load('top_model.pth'))
    model.eval()

    # Сохраняем модель
    torch.save(model.state_dict(), 'linreg_torch.pth')


class LogisticRegression(nn.Module):
    def __init__(self, in_features, classes):
        super().__init__()
        self.linear = nn.Linear(in_features, classes)

    def forward(self, x):
        return self.linear(x)


def draw_confusion_matrix(y_true, y_pred, classes):
    """
    Визуализация confusion matrix
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()


def logistic_reg():
    """
    Расширенная версия логистической регрессии
    """
    # Данные для многоклассовой классификации
    classes = 3
    X, y = make_classification_data(n=200, classes=classes)

    # Создаём датасет и даталоадер
    dataset = ClassificationDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f'Размер датасета: {len(dataset)}')
    print(f'Количество батчей: {len(dataloader)}')

    # Создаём модель, функцию потерь и оптимизатор
    model = LogisticRegression(in_features=2, classes=classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    # Обучаем модель
    epochs = 100
    all_y_true = []
    all_y_pred = []
    all_y_score = []

    for epoch in range(1, epochs + 1):
        total_loss = 0
        total_acc = 0
        batch_y_true = []
        batch_y_pred = []
        batch_y_score = []

        for i, (batch_X, batch_y) in enumerate(dataloader):
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            # Вычисляем метрики
            y_prob = torch.softmax(logits, dim=1)
            _, y_pred = torch.max(y_prob, 1)

            total_loss += loss.item()
            total_acc += (y_pred == batch_y).float().mean().item()

            batch_y_true.extend(batch_y.cpu().numpy())
            batch_y_pred.extend(y_pred.cpu().numpy())
            batch_y_score.extend(y_prob.cpu().detach().numpy())

        # Собираем метрики для всей эпохи
        all_y_true.extend(batch_y_true)
        all_y_pred.extend(batch_y_pred)
        all_y_score.extend(batch_y_score)

        avg_loss = total_loss / (i + 1)
        avg_acc = total_acc / (i + 1)
        precision = precision_score(batch_y_true, batch_y_pred, average='macro')
        recall = recall_score(batch_y_true, batch_y_pred, average='macro')
        f1 = f1_score(batch_y_true, batch_y_pred, average='macro')

        if epoch % 10 == 0:
            log_epoch(epoch, avg_loss, acc=avg_acc, precision=precision, recall=recall, f1=f1)

    # Вычисляем ROC-AUC
    if classes == 2:
        roc_auc = roc_auc_score(all_y_true, [score[1] for score in all_y_score])
        print(f'\nROC-AUC: {roc_auc:.4f}')

    # Визуализация
    draw_confusion_matrix(all_y_true, all_y_pred, classes=range(classes))

    # Сохраняем модель
    torch.save(model.state_dict(), 'logreg_torch.pth')


if __name__ == '__main__':
    linear_reg()
    logistic_reg()
