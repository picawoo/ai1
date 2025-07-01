import torch

def tensor_creation():
    """
    1.1 Создание тензоров
    Создайте следующие тензоры:
    - Тензор размером 3x4, заполненный случайными числами от 0 до 1
    - Тензор размером 2x3x4, заполненный нулями
    - Тензор размером 5x5, заполненный единицами
    - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)
    """
    random_tensor = torch.rand(3, 4)
    print(f'Рандомный тензор: {random_tensor}')
    zeros_tensor = torch.zeros(2, 3, 4)
    print(f'Тензор из нулей: {zeros_tensor}')
    ones_tensor = torch.ones(5, 5)
    print(f'Тензор из единиц: {ones_tensor}')
    range_tensor = torch.arange(0, 16).reshape(4, 4)
    print(f'Тензор 4 на 4: {range_tensor}')

def tensor_operations():
    """
    1.2 Операции с тензорами
    Дано: тензор A размером 3x4 и тензор B размером 4x3
    Выполните:
    - Транспонирование тензора A
    - Матричное умножение A и B
    - Поэлементное умножение A и транспонированного B
    - Вычислите сумму всех элементов тензора A
    """
    A = torch.arange(1, 13).reshape(3, 4)
    B = torch.arange(1, 13).reshape(4, 3)
    A_trans = A.T
    print(f'Транспонированный А: {A_trans}')
    matrix_multiplication = A @ B
    print(f'А на В: {matrix_multiplication}')
    multiplication_one_by_one = A * B.T
    print(f'Поэлементное умножение: {multiplication_one_by_one}')
    A_sum = torch.sum(A).item()
    print(f'Сумма элементов А: {A_sum}')

def tensor_slices():
    """
    1.3 Индексация и срезы
    Создайте тензор размером 5x5x5
    Извлеките:
    - Первую строку
    - Последний столбец
    - Подматрицу размером 2x2 из центра тензора
    - Все элементы с четными индексами
    """
    slice_tensor = torch.arange(1, 126).reshape(5, 5, 5)
    first_row = slice_tensor[0][0]
    print(f'Первая строка: {first_row}')
    last_column = slice_tensor[:, :, -1]
    print(f'Последний столбец: {last_column}')
    centre = slice_tensor[2, 2:4, 2:4] # кол-во столбцов и строк нечётно - взята подматрица с первым элементом в центре тензора
    print(f'Подматрица 2х2: {centre}')
    even_elems = slice_tensor[::2, ::2, ::2]
    print(f'Элементы с чётными индексами (по всем измерениям): {even_elems}')

def tensor_shapes():
    """
    1.4 Работа с формами
    Создайте тензор размером 24 элемента
    Преобразуйте его в формы:
    - 2x12
    - 3x8
    - 4x6
    - 2x3x4
    - 2x2x2x3
    """
    shape_tensor = torch.arange(24)
    shapes = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]
    reshaped_tensors = []
    for shape in shapes:
        res = shape_tensor.reshape(shape)
        reshaped_tensors.append(res)
        print(f'Тензор формы {shape}: {res}')

if __name__ == '__main__':
    tensor_creation()
    tensor_operations()
    tensor_slices()
    tensor_shapes()
