import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class Trainer:
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 tokenizer: Tokenizer,
                 device: torch.device,
                 lr: float = 1e-4,
                 max_grad_norm: float = 1.0):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.token_to_id("[PAD]"))

        os.makedirs("checkpoints", exist_ok=True)

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in progress_bar:
            inputs = batch.to(self.device)

            targets = inputs[:, 1:].contiguous()
            inputs = inputs[:, :-1].contiguous()

            self.optimizer.zero_grad()

            # forward pass
            outputs = self.model(inputs)

            # loss
            loss = self.criterion(
                outputs.view(-1, outputs.size(-1)),
                targets.view(-1)
            )

            # backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(self.train_loader)

    def validate(self) -> float:
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                inputs = batch.to(self.device)
                targets = inputs[:, 1:].contiguous()
                inputs = inputs[:, :-1].contiguous()

                outputs = self.model(inputs)
                loss = self.criterion(
                    outputs.view(-1, outputs.size(-1)),
                    targets.view(-1)
                )
                total_loss += loss.item()

        return total_loss / len(self.val_loader)

    def save_checkpoint(self, epoch: int, val_loss: float) -> None:
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'tokenizer': self.tokenizer.to_str()
        }

        path = f"checkpoints/model_epoch_{epoch}.pt"
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")

    def train(self, num_epochs: int = 10) -> None:
        best_val_loss = float('inf')

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate()

            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
