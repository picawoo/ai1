import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from torch.utils.data import Dataset, DataLoader
from dataset import TextDataset
from transformer import DecoderOnlyTransformer
from train import Trainer
import os
from tqdm import tqdm


def main():
    config = {
        'd_model': 512,
        'num_heads': 8,
        'num_layers': 6,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_length': 192,
        'batch_size': 1,
        'lr': 1e-4,
        'num_epochs': 3,
        'data_dir': 'texts'
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    file_paths = [os.path.join(config['data_dir'], f)
                  for f in os.listdir(config['data_dir'])
                  if f.endswith('.txt')]

    trainer = trainers.BpeTrainer(
        special_tokens=["[PAD]", "[BOS]", "[EOS]"],
        min_frequency=2
    )
    tokenizer.train(files=file_paths, trainer=trainer)
    tokenizer.save("tokenizer.json")

    train_dataset = TextDataset(
        file_paths=file_paths,
        tokenizer=tokenizer,
        max_length=config['max_length']
    )

    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_len=config['max_length'],
        pad_idx=tokenizer.token_to_id("[PAD]")
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        lr=config['lr']
    )

    trainer.train(num_epochs=config['num_epochs'])


def chat():
    """Интерактивный чат с обученной моделью"""
    # Загрузка токенизатора
    tokenizer = Tokenizer.from_file("tokenizer.json")

    # Загрузка конфигурации из чекпоинта
    checkpoint_path = "checkpoints/model_epoch_3.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # Получаем конфигурацию из чекпоинта или используем значения по умолчанию
        config = {
            'd_model': checkpoint.get('d_model', 512),
            'num_heads': checkpoint.get('num_heads', 8),
            'num_layers': checkpoint.get('num_layers', 6),
            'd_ff': checkpoint.get('d_ff', 2048),
            'dropout': checkpoint.get('dropout', 0.1),
            'max_len': checkpoint.get('max_len', 192),  # Используем max_len из чекпоинта
            'pad_idx': tokenizer.token_to_id("[PAD]")
        }

        model = DecoderOnlyTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            **config
        )

        # Загрузка весов с обработкой несоответствия размеров
        state_dict = checkpoint['model_state_dict']

        # Корректируем размер позиционного кодирования, если нужно
        if 'pos_encoding.pe' in state_dict:
            current_pe_size = model.pos_encoding.pe.size(1)
            saved_pe_size = state_dict['pos_encoding.pe'].size(1)

            if current_pe_size > saved_pe_size:
                # Дополняем нулями если текущий размер больше
                padding = torch.zeros(1, current_pe_size - saved_pe_size, config['d_model'])
                state_dict['pos_encoding.pe'] = torch.cat([
                    state_dict['pos_encoding.pe'],
                    padding
                ], dim=1)
            elif current_pe_size < saved_pe_size:
                # Обрезаем если текущий размер меньше
                state_dict['pos_encoding.pe'] = state_dict['pos_encoding.pe'][:, :current_pe_size, :]

        model.load_state_dict(state_dict, strict=False)
        print(f"Модель загружена из {checkpoint_path}")
    else:
        raise FileNotFoundError(f"Чекпоинт {checkpoint_path} не найден")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("\nЧат с Transformer моделью (введите 'quit' для выхода)")

    # Параметры генерации по умолчанию
    gen_config = {
        'max_length': 200,
        'temperature': 0.7,
        'top_k': 50,
        'use_beam_search': False,
        'beam_width': 3,
        'length_penalty': 0.6
    }

    while True:
        try:
            user_input = input("\nВы: ").strip()

            if user_input.lower() == 'quit':
                break
            if not user_input:
                continue

            # Токенизация ввода
            input_ids = [tokenizer.token_to_id("[BOS]")] + tokenizer.encode(user_input).ids
            input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

            # Генерация ответа
            with torch.no_grad():
                output_ids = model.generate(
                    input_tensor,
                    max_length=gen_config['max_length'],
                    temperature=gen_config['temperature'],
                    top_k=gen_config['top_k'],
                    use_beam_search=gen_config['use_beam_search'],
                    beam_width=gen_config['beam_width'],
                    length_penalty=gen_config['length_penalty']
                )

            # Декодирование и очистка ответа
            response = tokenizer.decode(output_ids[0].cpu().numpy().tolist())
            response = response.replace(user_input, "").strip()
            eos_pos = response.find("[EOS]")
            if eos_pos != -1:
                response = response[:eos_pos]

            print(f"Бот: {response}")

        except KeyboardInterrupt:
            print("\nЗавершение чата...")
            break
        except Exception as e:
            print(f"Ошибка: {str(e)}")
            continue


if __name__ == "__main__":
    # main()
    chat()
