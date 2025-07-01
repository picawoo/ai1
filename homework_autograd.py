import torch


def simple_grad():
    """
    2.1 Простые вычисления с градиентами
    Создайте тензоры x, y, z с requires_grad=True
    Вычислите функцию: f(x,y,z) = x^2 + y^2 + z^2 + 2*x*y*z
    Найдите градиенты по всем переменным
    Проверьте результат аналитически
    """
    x_value = 2.0
    y_value = 5.0
    z_value = 10.0

    x = torch.tensor(x_value, requires_grad=True)
    y = torch.tensor(y_value, requires_grad=True)
    z = torch.tensor(z_value, requires_grad=True)

    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
    f.backward()
    print(f'Функция: {f.item()}')
    print(f'Градиент по x: {x.grad.item()}')
    print(f'Градиент по y: {y.grad.item()}')
    print(f'Градиент по z: {z.grad.item()}')

    print(f'Проверка по х: {2 * x_value + 2 * y_value * z_value}')
    print(f'Проверка по y: {2 * y_value + 2 * x_value * z_value}')
    print(f'Проверка по z: {2 * z_value + 2 * x_value * y_value}')


def mse_grad():
    """
    2.2 Градиент функции потерь
    Реализуйте функцию MSE (Mean Squared Error):
    MSE = (1/n) * Σ(y_pred - y_true)^2
    где y_pred = w * x + b (линейная функция)
    Найдите градиенты по w и b
    """

    def MSE(x, y_true, w, b):
        """
        Вычисляет результат функции MSE
        :param x: входные данные
        :param y_true: истинные значения
        :param w: вес
        :param b: смещение
        :return: результат вычисления функции MSE
        """
        y_pred = w * x + b
        res = torch.mean((y_pred - y_true) ** 2)
        return res

    x = torch.tensor([1.0, 2.0, 3.0])
    y_true = torch.tensor([3.0, 6.0, 9.0])
    w = torch.tensor(1.0, requires_grad=True)
    b = torch.tensor(0.0, requires_grad=True)

    loss = MSE(x, y_true, w, b)
    loss.backward()

    print(f'MSE: {loss.item()}')
    print(f'Градиент по w: {w.grad.item()}')
    print(f'Градиент по b: {b.grad.item()}')


def chain_grad():
    """
    2.3 Цепное правило
    Реализуйте составную функцию: f(x) = sin(x^2 + 1)
    Найдите градиент df/dx
    Проверьте результат с помощью torch.autograd.grad
    """
    x = torch.tensor(2.0, requires_grad=True)

    def func(x):
        """
        Вычисляет результат составной функции f(x) = sin(x^2 + 1)
        :param x: входные данные
        :return: результат вычисления функции
        """
        return torch.sin(x ** 2 + 1)

    f = func(x)
    f.backward(retain_graph=True)
    backward_res = x.grad.item()
    autograd_res = torch.autograd.grad(outputs=f, inputs=x)[0].item()

    print(backward_res)
    print(autograd_res)


if __name__ == '__main__':
    simple_grad()
    mse_grad()
    chain_grad()
