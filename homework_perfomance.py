import torch
import time
from tabulate import tabulate


def create_tensors():
    """
    Создаёт тензоры по условию
    :return: кортеж из списков тензоров
    """
    cpu_tensors = [
        torch.rand(64, 1024, 1024),
        torch.rand(128, 512, 512),
        torch.rand(256, 256, 256)
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        gpu_tensors = [t.to(device) for t in cpu_tensors]
    else:
        gpu_tensors = None

    return cpu_tensors, gpu_tensors


cpu_tensors, gpu_tensors = create_tensors()


class Timer:
    def __init__(self, device='cpu'):
        self.device = device

    def __enter__(self):
        if self.device == 'cuda' and torch.cuda.is_available():
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.start_time = time.time()
        return self

    def __exit__(self, *args):
        if self.device == 'cuda' and torch.cuda.is_available():
            self.end_event.record()
            torch.cuda.synchronize()
            self.elapsed = self.start_event.elapsed_time(self.end_event) / 1000
        else:
            self.elapsed = time.time() - self.start_time


def benchmark():
    """
    Производит замеры и печатает таблицу с результатами
    """
    operations = [
        ('Матричное умножение', lambda a: torch.matmul(a, a.transpose(1, 2))),
        ('Поэлементное сложение', lambda a: a + a),
        ('Поэлементное умножение', lambda a: a * a),
        ('Транспонирование', lambda a: a.transpose(1, 2)),
        ('Сумма элементов', lambda a: a.sum())
    ]

    results = []
    for name, operation in operations:
        row = {'Операция': name}
        tensor = cpu_tensors[2]
        with Timer(device='cpu') as t:
            _ = operation(tensor)
        row['CPU (сек)'] = f"{t.elapsed:.6f}"
        if gpu_tensors:
            tensor_gpu = gpu_tensors[2]
            for _ in range(10):  # разогрев
                _ = operation(tensor_gpu)
            with Timer(device='cuda') as t:
                _ = operation(tensor_gpu)
            row['GPU (сек)'] = f"{t.elapsed:.6f}"
            row['Ускорение'] = f"{(float(row['CPU (сек)']) / t.elapsed):.2f}x"

        results.append(row)

    print(tabulate(results, headers="keys", tablefmt="grid", floatfmt=".6f"))


benchmark()
