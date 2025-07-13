import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.utils.data import Dataset, DataLoader
from layers import DecoderLayer, PositionalEncoding
import os
from tqdm import tqdm


class DecoderOnlyTransformer(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 d_model: int = 512,
                 num_heads: int = 8,
                 num_layers: int = 6,
                 d_ff: int = 2048,
                 dropout: float = 0.1,
                 max_len: int = 512,
                 pad_idx: int = 0):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Linear(d_model, vocab_size)

        self.max_len = max_len
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Embed tokens and add positional encoding
        x = self.token_embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.norm(x)
        x = self.proj(x)
        return x

    def beam_search(
            self,
            input_ids: torch.Tensor,
            beam_width: int = 5,
            max_length: int = 100,
            temperature: float = 1.0,
            length_penalty: float = 0.6,
            early_stopping: bool = True
    ) -> torch.Tensor:
        self.eval()
        device = input_ids.device
        batch_size = input_ids.size(0)

        if batch_size != 1:
            raise ValueError("Beam search поддерживает только batch_size = 1")

        # Инициализация лучей
        beam_scores = torch.zeros(beam_width, device=device)
        beam_sequences = input_ids.clone()
        beam_sequences = beam_sequences.unsqueeze(1).repeat(1, beam_width, 1)

        # Словарь завершенных гипотез
        completed_hypotheses = []

        for step in range(max_length):
            # Получаем предсказания для всех гипотез
            with torch.no_grad():
                outputs = self(beam_sequences.view(-1, beam_sequences.size(-1)))
                next_token_logits = outputs[:, -1, :] / temperature
                next_token_probs = F.softmax(next_token_logits, dim=-1)

            # Вычисляем общие вероятности
            vocab_size = next_token_probs.size(-1)
            if step == 0:
                # На первом шаге просто берем топ-K токенов
                scores = next_token_probs.log()
                topk_scores, topk_indices = scores.topk(beam_width, dim=-1)

                # Обновляем последовательности
                beam_sequences = torch.cat([
                    beam_sequences.expand(-1, beam_width, -1),
                    topk_indices.unsqueeze(-1)
                ], dim=-1)
                beam_scores = topk_scores[0]
            else:
                # Комбинируем предыдущие вероятности с новыми
                scores = next_token_probs.log() + beam_scores.unsqueeze(-1)
                scores = scores.view(-1)

                # Выбираем топ-K комбинаций
                topk_scores, topk_indices = scores.topk(beam_width, dim=-1)

                # Восстанавливаем индексы лучей и токенов
                beam_indices = topk_indices // vocab_size
                token_indices = topk_indices % vocab_size

                # Обновляем последовательности
                beam_sequences = torch.cat([
                    beam_sequences[0, beam_indices],
                    token_indices.unsqueeze(-1)
                ], dim=-1).unsqueeze(0)
                beam_scores = topk_scores

            # Проверяем завершенные гипотезы
            eos_mask = (beam_sequences[0, :, -1] == self.pad_idx)
            if eos_mask.any():
                # Добавляем завершенные гипотезы
                for i in torch.where(eos_mask)[0]:
                    sequence = beam_sequences[0, i]
                    score = beam_scores[i] / (sequence.size(0) ** length_penalty)
                    completed_hypotheses.append((sequence, score))

                    # Оставляем только незавершенные гипотезы
                    active_mask = ~eos_mask
                    if active_mask.any():
                        beam_sequences = beam_sequences[:, active_mask]
                        beam_scores = beam_scores[active_mask]
                        beam_width = beam_scores.size(0)
                    else:
                        break

                if early_stopping and len(completed_hypotheses) >= beam_width:
                    break

        # Добавляем оставшиеся гипотезы, если не было early_stopping
        if not early_stopping or not completed_hypotheses:
            for i in range(beam_sequences.size(1)):
                sequence = beam_sequences[0, i]
                score = beam_scores[i] / (sequence.size(0) ** length_penalty)
                completed_hypotheses.append((sequence, score))

                # Сортируем гипотезы по score и выбираем лучшую
                completed_hypotheses.sort(key=lambda x: x[1], reverse=True)
                best_sequence = completed_hypotheses[0][0].unsqueeze(0)

        return best_sequence

    def generate(
            self,
            input_ids: torch.Tensor,
            max_length: int = 100,
            temperature: float = 1.0,
            top_k: int = 50,
            use_beam_search: bool = False,
            beam_width: int = 5,
            length_penalty: float = 0.6
    ) -> torch.Tensor:
        if use_beam_search:
            return self.beam_search(
                input_ids,
                beam_width=beam_width,
                max_length=max_length,
                temperature=temperature,
                length_penalty=length_penalty
            )
        else:
            for _ in range(max_length):
                outputs = self(input_ids[:, -self.max_len:])
                next_token_logits = outputs[:, -1, :] / temperature

                if top_k > 0:
                    top_k_values, _ = torch.topk(next_token_logits, top_k)
                    next_token_logits[next_token_logits < top_k_values[:, [-1]]] = float('-inf')

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if (next_token == self.pad_idx).all():
                    break
            return input_ids
