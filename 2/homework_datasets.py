import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import homework_model_modification as mod


class CustomDataset(Dataset):
    def __init__(self, path_file, numeric_columns: list, string_columns: list, binary_columns: list,
                 target_column: str):
        self.path_file = path_file
        self.numeric_columns = numeric_columns
        self.string_columns = string_columns
        self.binary_columns = binary_columns
        self.target_column = target_column
        self.label_encoders = {}
        self.one_hot_encoders = {}

        self.df = pd.read_csv(path_file)
        self.df = self.df.dropna()

        self._convert_numeric()
        self._convert_string()
        self._convert_binary()

    def _convert_numeric(self):
        if self.numeric_columns:
            scaler = StandardScaler()
            self.df[self.numeric_columns] = scaler.fit_transform(self.df[self.numeric_columns])

    def _convert_string(self):
        if self.string_columns:
            for column in self.string_columns:
                le = LabelEncoder()
                self.df[column] = self.df[column].fillna('missing')
                self.df[column] = le.fit_transform(self.df[column])
                self.label_encoders[column] = le

    def _convert_binary(self):
        if self.binary_columns:
            for column in self.binary_columns:
                encoder = OneHotEncoder(sparse_output=False, drop='first')
                transformed = encoder.fit_transform(self.df[[column]])
                new_columns = [f"{column}_{cat}" for cat in encoder.categories_[0][1:]]
                self.df[new_columns] = transformed
                self.df = self.df.drop(column, axis=1)
                self.one_hot_encoders[column] = encoder

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        data_one_row = self.df.iloc[index]
        train = data_one_row.drop(self.target_column).values.astype(np.float32)
        target = np.array(data_one_row[self.target_column], dtype=np.float32)
        return torch.tensor(train), torch.tensor(target)


def learn(dataset):
    batch_size: int = 32,
    num_epochs: int = 100,
    learning_rate: float = 0.001,
    l1_lambda: float = 0.01,
    patience: int = 5,
    validation_split: float = 0.2

    in_features = len(fdataset.df.columns) - 1  # Все колонки, кроме целевой
    out_features = 1

    model = mod.LinearModel(in_features, out_features)
    criterion = nn.MSELoss()  # Функция потерь для регрессии
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)

    # Разделяем данные на обучающую и валидационную выборки
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    train_indices, val_indices = train_test_split(
        indices, test_size=validation_split, random_state=42
    )

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=32, sampler=val_sampler)

    early_stopping = mod.EarlyStopping(patience=patience, delta=0)

    # Обучение
    print("Starting training...")
    for epoch in range(100):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(dtype=torch.float32), target.to(dtype=torch.float32)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(dtype=torch.float32), target.to(dtype=torch.float32)
                output = model(data)
                loss = criterion(output, target.view(-1, 1))
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")


if __name__ == '__main__':
    fdataset = CustomDataset(path_file="csv/diabetes.csv",
                             numeric_columns=["pregnancies", "glucose", "diastolic", "triceps",
                                              "insulin", "bmi", "dpf", "age"],
                             string_columns=[],
                             binary_columns=[],
                             target_column="diabetes")

    sdataset = CustomDataset(path_file="csv/titanic.csv",
                             numeric_columns=["Age", "Fare", "Pclass", "SibSp", "Parch"],
                             string_columns=["Name", "Embarked", "Cabin", "Ticket"],
                             binary_columns=["Sex"],
                             target_column="Survived")

    learn(fdataset)
    learn(sdataset)
