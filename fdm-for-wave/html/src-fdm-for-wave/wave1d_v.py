# -*- coding: utf-8 -*-

"""
Решение одномерного волнового уравнения с однородными 
граничными условиями Дирихле.
Функция solver допускает скалярную и векторизованную версию.
Эта функция принимает дополнительный аргумент "version":
version='scalar' использует циклы по узлам сетки, а 
version='vectorized' использует векторизованные вычисления.
"""
import numpy as np

def solver(I, V, f, c, l, tau, gamma, T, user_action=None,
           version='vectorized'):
    """Решает u_tt=c^2*u_xx + f на (0,l)x(0,T]."""
    K = int(round(T/tau))
    t = np.linspace(0, K*tau, K+1)   # Сетка по времени
    dx = tau*c/float(gamma)
    N = int(round(l/dx))
    x = np.linspace(0, l, N+1)       # Сетка по пространству
    C2 = gamma**2                         # Вспомогательная переменная
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)

    y   = np.zeros(N+1)   # Массив с решением на новом слое n+1
    y_1 = np.zeros(N+1)   # Решение на пердыдущем слое n
    y_2 = np.zeros(N+1)   # Решенеие на слое т-1

    import time;  t0 = time.clock()  # для измеренеия процессорного времени

    # Задаем начальные условия
    for i in range(0,N+1):
        y_1[i] = I(x[i])

    if user_action is not None:
        user_action(y_1, x, t, 0)

    # Используем специальную формулу для первого временного слоя
    n = 0
    for i in range(1, N):
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*C2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
               0.5*tau**2*f(x[i], t[n])
    y[0] = 0;  y[N] = 0

    if user_action is not None:
        user_action(y, x, t, 1)

    # Изменяем переменные перед переходом на следующий
    # временной слой
    y_2[:] = y_1;  y_1[:] = y

    for n in range(1, K):
        # Пересчитываем значения во внутренних узлах сетки на слое n+1

        if version == 'scalar':
            for i in range(1, N):
                y[i] = - y_2[i] + 2*y_1[i] + \
                       C2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
                       tau**2*f(x[i], t[n])
        elif version == 'vectorized':   # тип срезов (1:-1)
            f_a = f(x, t[n])
            y[1:-1] = - y_2[1:-1] + 2*y_1[1:-1] + \
                C2*(y_1[0:-2] - 2*y_1[1:-1] + y_1[2:]) + \
                tau**2*f_a[1:-1]
        elif version == 'vectorized2':  # тип срезов (1:N)
            f_a = f(x, t[n])
            y[1:N] =  - y_2[1:N] + 2*y_1[1:N] + \
                C2*(y_1[0:N-1] - 2*y_1[1:N] + y_1[2:N+1]) + \
                tau**2*f_a[1:N]

        # Задаем граничные условия
        y[0] = 0;  y[N] = 0
        if user_action is not None:
            if user_action(y, x, t, n+1):
                break

        # Изменяем переменные перед переходом на следующий
        y_2[:] = y_1;  y_1[:] = y

    cpu_time = t0 - time.clock()
    return y, x, t, cpu_time

def viz(
    I, V, f, c, l, tau, gamma, T,  # Параметры задачи
    umin, umax,               # Интервал для отображения u
    animate=True,             # Расчет с анимацией?
    tool='matplotlib',        # 'matplotlib' или 'scitools'
    solver_function=solver,   # Функция, реализующая алгоритм
    version='vectorized',     # 'scalar' или 'vectorized'
    ):
    import wave1d_1
    if version == 'vectorized':
        # Повторно использует viz из wave1d_1, но с новой 
        # векторизованной функцией solver из данного модуля
        # (где version='vectorized' задан по умолчанию;
        # wave1d_1.viz не имеет этого аргумента)
        cpu = wave1d_1.viz(
            I, V, f, c, l, tau, gamma, T, umin, umax,
            animate, tool, solver_function=solver)
    elif version == 'scalar':
        # Вызваем wave1d_1.viz со скалярным солвером
        # и используем wave1d_1.solver.
        cpu = wave1d_1.viz(
            I, V, f, c, l, tau, gamma, T, umin, umax,
            animate, tool,
            solver_function=wave1d_1.solver)
        # --- Solution 2 ---
        # Решение 2: "обернуть" solver данного модуля 
        # используя functools.partial
        #scalar_solver = functools.partial(solver, version='scalar')
    return cpu

def test_quadratic():
    """
    Проверяет воспроизводят ли скалярная и векторизованная версии
    решение u(x,t)=x(l-x)(1+t/2) точно.
    """
    # Следующие функции должны работать при x заданном как массив или скаляр
    u_exact = lambda x, t: x*(l - x)*(1 + 0.5*t)
    I = lambda x: u_exact(x, 0)
    V = lambda x: 0.5*u_exact(x, 0)
    # f --- скаляр (zeros_like(x) тоже работает для скалярного x)
    f = lambda x, t: np.zeros_like(x) + 2*c**2*(1 + 0.5*t)

    l = 2.5
    c = 1.5
    gamma = 0.75
    N = 3  # Очень грубая сетка для теста
    tau = gamma*(l/N)/c
    T = 18

    def assert_no_error(y, x, t, n):
        u_e = u_exact(x, t[n])
        tol = 1E-13
        diff = np.abs(y - u_e).max()
        assert diff < tol

    solver(I, V, f, c, l, tau, gamma, T,
           user_action=assert_no_error, version='scalar')
    solver(I, V, f, c, l, tau, gamma, T,
           user_action=assert_no_error, version='vectorized')

def guitar(gamma):
    """Треугольная волна"""
    l = 0.75
    x0 = 0.8*l
    a = 0.005
    freq = 440
    wavelength = 2*l
    c = freq*wavelength
    omega = 2*pi*freq
    num_periods = 1
    T = 2*pi/omega*num_periods
    # Выбираем tau таким же как при условии устойчивости для N=50
    tau = l/50./c

    def I(x):
        return a*x/x0 if x < x0 else a/(l-x0)*(l-x)

    umin = -1.2*a;  umax = -umin
    cpu = viz(I, 0, 0, c, l, tau, gamma, T, umin, umax, animate=True)

def run_efficiency_experiments():
    l = 1
    x0 = 0.8*l
    a = 1
    c = 2
    T = 8
    gamma = 0.9
    umin = -1.2*a;  umax = -umin

    def I(x):
        return a*x/x0 if x < x0 else a/(l-x0)*(l-x)

    intervals = []
    speedup = []
    for N in [50, 100, 200, 400, 800]:
        dx = float(l)/N
        tau = gamma/c*dx
        print 'solving scalar N=%d' % N,
        cpu_s = viz(I, 0, 0, c, l, tau, gamma, T, umin, umax,
                    animate=False, version='scalar')
        print cpu_s
        print 'solving vectorized N=%d' % N,
        cpu_v = viz(I, 0, 0, c, l, tau, gamma, T, umin, umax,
                    animate=False, version='vectorized')
        print cpu_v
        intervals.append(N)
        speedup.append(cpu_s/float(cpu_v))
        print 'N=%3d: cpu_v/cpu_s: %.3f' % (N, 1./speedup[-1])
    print 'N:', intervals
    print 'Увеличение скорости расчета:', speedup

if __name__ == '__main__':
    test_quadratic()
    import sys
    try:
        gamma = float(sys.argv[1])
        print 'gamma=%g' % gamma
    except IndexError:
        gamma = 0.85
    guitar(gamma)
    #run_efficiency_experiments()
