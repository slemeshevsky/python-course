# -*- coding: utf-8 -*-

#!/usr/bin/env python
"""
Решение одномерного волнового уравнения с однородными 
граничными условиями Неймана

  y, x, t, cpu = solver(I, V, f, c, l, tau, gamma, T, user_action)

Функция solver решает волновое уравнение

   u_tt = c**2*u_xx + f(x,t) на (0,l) 
дополненное граничными условиями du/dn=0 на x=0 и x = l.

tau --- шаг сетки по времени
T --- конечный момент времени
gamma --- число Куранта (=c*tau/h).
h --- шаг сетки по пространству, полученный с учетом tau и gamma.

I и f функции: I(x), f(x,t).
user_action функция от (y, x, t, n), в которой вызываемый код
моджет добавить визуализацию, вычисление погрешности, анализ данных,
сохрранение решения на диск и т.д.

Функция viz::

  viz(I, V, f, c, l, tau, gamma, T, umin, umax, animate=True)

вызвает функции solver и user_action, которая может строить график 
решения на экране (как анимацию)
"""
import numpy as np

def solver(I, V, f, c, l, tau, gamma, T, user_action=None):
    """
    Решаем u_tt=c^2*u_xx + f on (0,l)x(0,T].
    """
    K = int(round(T/tau))
    t = np.linspace(0, K*tau, K+1)   # Сетка по времени
    h = tau*c/float(gamma)
    N = int(round(l/h))
    x = np.linspace(0, l, N+1)       # Сетка по пространству
    gamma2 = gamma**2; tau2 = tau*tau            # Вспомогательные переменные

    # Wrap user-given f, V
    if f is None or f == 0:
        f = (lambda x, t: 0)
    if V is None or V == 0:
        V = (lambda x: 0)

    y   = np.zeros(N+3)   # Массив с решением на новом слое n+1
    y_1 = np.zeros(N+3)   # Решение на пердыдущем слое n
    y_2 = np.zeros(N+3)   # Решенеие на слое n-1

    Ix = range(1, u.shape[0]-1)
    It = range(0, t.shape[0])

    import time;  t0 = time.clock()  # для измерения процессорного времени

    # Задаем начальные условия
    for i in Ix:
        y_1[i] = I(x[i-Ix[0]])  # Отметьте преобразование индексов по x
    # Мнимые значения устанавливаем в соответствии с du/dx=0
    i = Ix[0]
    y_1[i-1] = y_1[i+1]
    i = Ix[-1]
    y_1[i+1] = y_1[i-1]

    if user_action is not None:
        # Удостоверяемся, что передаем часть массива y, которая соответствует x
        user_action(y_1[Ix[0]:Ix[-1]+1], x, t, 0)

    # Используем специальную формулу для первого временного слоя
    # --- Start Example 2 ---
    for i in Ix:
        y[i] = y_1[i] + tau*V(x[i-Ix[0]]) + \
               0.5*gamma2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
               0.5*tau2*f(x[i-Ix[0]], t[0])
    # --- End Example 2 ---
    # Мнимые значения устанавливаем в соответствии с du/dx=0
    i = Ix[0]
    y[i-1] = y[i+1]
    i = Ix[-1]
    y[i+1] = y[i-1]

    if user_action is not None:
        # Удостоверяемся, что передаем часть массива y, которая соответствует x
        user_action(y[Ix[0]:Ix[-1]+1], x, t, 1)

    # Изменяем переменные перед переходом на следующий
	# временной слой
    #y_2[:] = y_1;  y_1[:] = y  # более безопасно, но медленнее
    y_2, y_1, y = y_1, y, y_2

    for n in range(1, K):
	    # --- Start Example 1 ---
        for i in Ix:
            y[i] = - y_2[i] + 2*y_1[i] + \
                   gamma2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
                   tau2*f(x[i-Ix[0]], t[n])
	    # --- End Example 1 ---

	    # --- Start Example 3 ---
        # Мнимые значения устанавливаем в соответствии с du/dx=0
        i = Ix[0]
        y[i-1] = y[i+1]
        i = Ix[-1]
        y[i+1] = y[i-1]
        # --- End Example 3 ---

        if user_action is not None:
            # Удостоверяемся, что передаем часть массива y, которая соответствует x
            if user_action(y[Ix[0]:Ix[-1]+1], x, t, n+1):
                break

        # Update data structures for next step
        # Изменяем переменные перед переходом на следующий
        # временной слой
        #y_2[:] = y_1;  y_1[:] = y  # более безопасно, но медленнее
        y_2, y_1, y = y_1, y, y_2

    # Неправильное присвоение y = y_2 должно
    # быть скорректировано перед возвратом
    y = y_1
    cpu_time = t0 - time.clock()
    return y[1:-1], x, t, cpu_time


from wave1D_u0 import viz
from wave1D_n0 import plug
# Не можем просто импортировать test_plug, так как wave1d_n.test_plug
# будет вызывать в этом случае wave1d.solver, а не солвер отсюду

def test_plug():
    """
	Проверяет, что начальная функция вновь появляется
	после одного периода, если gamma=1.
    """
    l = 1.0
    I = lambda x: 0 if abs(x-l/2.0) > 0.1 else 1

    N = 10
    c = 0.5
    gamma = 1
    tau = gamma*(l/N)/c
    nperiods = 4
    T = l/c*nperiods  # Один период: c*T = l
    y, x, t, cpu = solver(
        I=I, V=None, f=None, c=c, l=l,
        tau=tau, gamma=gamma, T=T, user_action=None)
    y_0 = np.array([I(x_) for x_ in x])
    diff = np.abs(y - y_0).max()
    tol = 1E-13
    assert diff < tol

if __name__ == '__main__':
    test_plug()
