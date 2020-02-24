# -*- coding: utf-8 -*-
def fib(n):
    """
    Возвращает список первых n чисел Фибоначи
    """
    f0, f1 = 0, 1
    f = [1]*n
    for i in range(1, n):
        f[i] = f0 + f1
        f0, f1 = f1, f[i]

    return f

print(fib(10))
