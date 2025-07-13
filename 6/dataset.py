import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.utils.data import Dataset, DataLoader
import os
from tqdm import tqdm


class TextDataset(Dataset):
    def __init__(self,
                 file_paths: list,
                 tokenizer: Tokenizer,
                 max_length: int = 512,
                 bos_token: str = "[BOS]",
                 eos_token: str = "[EOS]",
                 pad_token: str = "[PAD]"):
        self.tokenizer = tokenizer
        self.max_length = max_length

        if not all(t in tokenizer.get_vocab() for t in [bos_token, eos_token, pad_token]):
            tokenizer.add_special_tokens([bos_token, eos_token, pad_token])

        self.bos_id = tokenizer.token_to_id(bos_token)
        self.eos_id = tokenizer.token_to_id(eos_token)
        self.pad_id = tokenizer.token_to_id(pad_token)

        # Чтение и токенизация
        self.data = []
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                tokens = tokenizer.encode(text).ids

                for i in range(0, len(tokens), max_length - 2):
                    chunk = tokens[i:i + max_length - 2]
                    chunk = [self.bos_id] + chunk + [self.eos_id]
                    if len(chunk) < max_length:
                        chunk += [self.pad_id] * (max_length - len(chunk))
                    self.data.append(chunk[:max_length])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.data[idx], dtype=torch.long)
