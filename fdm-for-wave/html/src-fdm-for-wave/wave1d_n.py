#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Решение одномерного волнового уравнения с однородными 
граничными условиями Неймана

  y, x, t, cpu = solver(I, V, f, c, ul, ur, l, tau, gamma, T, 
                        user_action, version)

Функция solver решает волновое уравнение

   u_tt = c**2*u_xx + f(x,t) на (0,l) 
с u=ul или du/dn=0 при x=0, и u=ur или du/dn=0 
при x = l. Если ul или ur равны None, используется условие
du/dn=0, иначе используеются условия Дирихле
с ul(t) и/или ur(t). Начальные условия: u=I(x), u_t=V(x).


tau --- шаг сетки по времени
T --- конечный момент времени
gamma --- число Куранта (=c*tau/h).
h --- шаг сетки по пространству, полученный с учетом tau и gamma.

I, f ul и ur --- функции: I(x), f(x,t), ul(t), ur(t). ul и ur 
могут также принимать значение 0, или None, где None соответствует
условию Неймана. f и V также могут принимать значения 0 или None 
(эквивалентно 0).

user_action функция от (y, x, t, n), в которой вызываемый код
может добавить визуализацию, вычисление погрешности, анализ данных,
сохрранение решения на диск и т.д.

Функция viz::

  viz(I, V, f, c, ul, ur, l, tau, gamma, T, umin, umax, 
      version='scalar', animate=True)

вызвает функции solver и user_action, которая может строить график 
решения на экране (как анимацию)
"""

import numpy as np
import scitools.std as plt

def solver(I, V, f, c, ul, ur, l, tau, gamma, T,
           user_action=None, version='scalar'):
    """
    Решает u_tt=c^2*u_xx + f на (0,l)x(0,T].
    u(0,t)=ul(t) или du/dn=0 (ul=None), u(l,t)=ur(t) или du/dn=0 (u_l=None).
    """
    K = int(round(T/tau))
    t = np.linspace(0, K*tau, K+1)   # Сетка по времени
    h = tau*c/float(gamma)
    N = int(round(l/h))
    x = np.linspace(0, l, N+1)       # Сетка по пространству
    gamma2 = gamma**2; tau2 = tau*tau            # Вспомогательные переменные

    # Обертка для заданных пользователем f, I, V, ul, ur
    if f is None or f == 0:
        f = (lambda x, t: 0) if version == 'scalar' else \
            lambda x, t: np.zeros(x.shape)
    if I is None or I == 0:
        I = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if V is None or V == 0:
        V = (lambda x: 0) if version == 'scalar' else \
            lambda x: np.zeros(x.shape)
    if ul is not None:
        if isinstance(ul, (float,int)) and ul == 0:
            ul = lambda t: 0
        # иначе: ul(t) --- функция
    if ur is not None:
        if isinstance(ur, (float,int)) and ur == 0:
            ur = lambda t: 0
        # иначе: ur(t) --- функция

    y   = np.zeros(N+1)   # Массив решения на новом слое
    y_1 = np.zeros(N+1)   # Решение на слое n
    y_2 = np.zeros(N+1)   # Решение на слое n-1

    Ix = range(0, N+1)
    It = range(0, K+1)

    import time;  t0 = time.clock()  # измерение процессорного времени

    # Задаем начальные условие в y_1
    for i in Ix:
        y_1[i] = I(x[i])

    if user_action is not None:
        user_action(y_1, x, t, 0)

    # Специальная формула для первого слоя
    for i in Ix[1:-1]:
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*gamma2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
               0.5*tau2*f(x[i], t[0])

    i = Ix[0]
    if ul is None:
        # Установка граничных условий du/dn = 0
        # x=0: i-1 -> i+1 так как y[i-1]=y[i+1]
        # x=l: i+1 -> i-1 так как y[i+1]=y[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*gamma2*(y_1[im1] - 2*y_1[i] + y_1[ip1]) + \
               0.5*tau2*f(x[i], t[0])
    else:
        y[0] = ul(tau)

    i = Ix[-1]
    if ur is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*gamma2*(y_1[im1] - 2*y_1[i] + y_1[ip1]) + \
               0.5*tau2*f(x[i], t[0])
    else:
        y[i] = ur(tau)

    if user_action is not None:
        user_action(y, x, t, 1)

    # Обновляем данные для следущего слоя
    #y_2[:] = y_1;  y_1[:] = y  # безопасно, но медленнее
    y_2, y_1, y = y_1, y, y_2

    for n in It[1:-1]:
        # Расчет во внутренних точках
        if version == 'scalar':
            for i in Ix[1:-1]:
                y[i] = - y_2[i] + 2*y_1[i] + \
                       gamma2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + \
                       tau2*f(x[i], t[n])

        elif version == 'vectorized':
            y[1:-1] = - y_2[1:-1] + 2*y_1[1:-1] + \
                      gamma2*(y_1[0:-2] - 2*y_1[1:-1] + y_1[2:]) + \
                      tau2*f(x[1:-1], t[n])
        else:
            raise ValueError('version=%s' % version)

        # Добавляем граничные условия
        i = Ix[0]
        if ul is None:
            # Устанавливаем граничные условия
            # x=0: i-1 -> i+1 так как y[i-1]=y[i+1] при du/dn=0
            # x=l: i+1 -> i-1 так как y[i+1]=y[i-1] при du/dn=0
            ip1 = i+1
            im1 = ip1
            y[i] = - y_2[i] + 2*y_1[i] + \
                   gamma2*(y_1[im1] - 2*y_1[i] + y_1[ip1]) + \
                   tau2*f(x[i], t[n])
        else:
            y[0] = ul(t[n+1])

        i = Ix[-1]
        if ur is None:
            im1 = i-1
            ip1 = im1
            y[i] = - y_2[i] + 2*y_1[i] + \
                   gamma2*(y_1[im1] - 2*y_1[i] + y_1[ip1]) + \
                   tau2*f(x[i], t[n])
        else:
            y[i] = ur(t[n+1])

        if user_action is not None:
            if user_action(y, x, t, n+1):
                break

        # Обновляем данные для следующего слоя
        #y_2[:] = y_1;  y_1[:] = y
        y_2, y_1, y = y_1, y, y_2

    y = y_1
    cpu_time = t0 - time.clock()
    return y, x, t, cpu_time


def viz(I, V, f, c, ul, ur, l, tau, gamma, T, umin, umax,
        version='scalar', animate=True):
    """Запуск солвера и визуализация y на каждом временном слое."""
    import time, glob, os
    if callable(ul):
        bc_left = 'u(0,t)=ul(t)'
    elif ul is None:
        bc_left = 'du(0,t)/h=0'
    else:
        bc_left = 'u(0,t)=0'
    if callable(ur):
        bc_right = 'u(l,t)=ur(t)'
    elif ur is None:
        bc_right = 'du(l,t)/h=0'
    else:
        bc_right = 'u(l,t)=0'

    def plot_u(y, x, t, n):
        """функция user_action для солвера."""
        # Работает только с scitools, см. wave1d_1.py для версии с matplotlib
        plt.plot(x, y, 'r-',
                 xlabel='x', ylabel='u',
                 axis=[0, l, umin, umax],
                 title='t=%.3f, %s, %s' % (t[n], bc_left, bc_right))
        # Пусть начальные условия остаются на экране в течение 2 секунд,
        # иначе пауза в 0.2 секунды между каждым графиком
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig('frame_%04d.png' % n)  # для создания видео

    # Очистка старых кадров видео
    for filename in glob.glob('frame_*.png'):
        os.remove(filename)

    user_action = plot_u if animate else None
    y, x, t, cpu = solver(I, V, f, c, ul, ur, l, tau, gamma, T,
                          user_action, version)
    if animate:
        plt.movie('frame_*.png', encoder='html', fps=4,
                  output_file='movie.html')
        # Создаем другие форматы видео: Flash, Webm, Ogg, MP4
        codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                         libtheora='ogg')
        fps = 6
        filespec = 'frame_%04d.png'
        movie_program = 'ffmpeg'  # или 'avconv'
        for codec in codec2ext:
            ext = codec2ext[codec]
            cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
                  '-vcodec %(codec)s movie.%(ext)s' % vars()
            print cmd
            os.system(cmd)
    return cpu

def test_constant():
    """
    Тестируем работу скалярной и векторизованой версий
    для постоянного u(x,t). Выполняем расчет на отрезке
    [0, l] и применяем условия Неймана и Дирихле на обоих 
    границах.
    """
    u_const = 0.45
    u_exact = lambda x, t: u_const
    I = lambda x: u_exact(x, 0)
    V = lambda x: 0
    f = lambda x, t: 0

    def assert_no_error(y, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(y - u_e).max()
        msg = 'diff=%E, t_%d=%g' % (diff, n, t[n])
        tol = 1E-13
        assert diff < tol, msg

    for ul in (None, lambda t: u_const):
        for ur in (None, lambda t: u_const):
            l = 2.5
            c = 1.5
            gamma = 0.75
            N = 3  # Очень грубая сетка для точного теста
            tau = gamma*(l/N)/c
            T = 18  

            solver(I, V, f, c, ul, ur, l, tau, gamma, T,
                   user_action=assert_no_error,
                   version='scalar')
            solver(I, V, f, c, ul, ur, l, tau, gamma, T,
                   user_action=assert_no_error,
                   version='vectorized')
            print ul, ur

def test_quadratic():
    """
    Тестируем работу скалярной и векторизованной версий для 
    u(x,t)=x(l-x)(1+t/2), которое воспроизводится точно.
    Расчет на отрезке [0, l].
    Отметим: использование симметричных условий на x=l/2
    (ul=None, l=l/2 в вызове солвера) воспроизводится неточно.
    """
    u_exact = lambda x, t: x*(l-x)*(1+0.5*t)
    I = lambda x: u_exact(x, 0)
    V = lambda x: 0.5*u_exact(x, 0)
    f = lambda x, t: 2*(1+0.5*t)*c**2
    ul = lambda t: u_exact(0, t)
    ur = None
    ur = 0
    l = 2.5
    c = 1.5
    gamma = 0.75
    N = 3
    tau = gamma*(l/N)/c
    T = 18

    def assert_no_error(y, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(y - u_e).max()
        msg = 'diff=%E, t_%d=%g' % (diff, n, t[n])
        tol = 1E-13
        assert diff < tol, msg

    solver(I, V, f, c, ul, ur, l, tau, gamma, T,
           user_action=assert_no_error, version='scalar')
    solver(I, V, f, c, ul, ur, l, tau, gamma, T,
           user_action=assert_no_error, version='vectorized')


def plug(gamma=1, N=50, animate=True, version='scalar', T=2, loc=0.5,
         bc_left='u=0', ic='u'):
    """Профиль-площадка в качестве начальных условий."""
    l = 1.
    c = 1

    def I(x):
        if abs(x-loc) > 0.1:
            return 0
        else:
            return 1

    u_l = 0 if bc_left == 'u=0' else None
    tau = (l/N)/c  
    if ic == 'u':
        # u(x,0)=plug, u_t(x,0)=0
        cpu = viz(lambda x: 0 if abs(x-loc) > 0.1 else 1,
                  None, None, c, u_l, None, l, tau, gamma, T,
                  umin=-1.1, umax=1.1, version=version, animate=animate)
    else:
        # u(x,0)=0, u_t(x,0)=plug
        cpu = viz(None, lambda x: 0 if abs(x-loc) > 0.1 else 1,
                  None, c, u_l, None, l, tau, gamma, T,
                  umin=-0.25, umax=0.25, version=version, animate=animate)

def gaussian(gamma=1, N=50, animate=True, version='scalar', T=1, loc=5,
             bc_left='u=0', ic='u'):
    """Функция Гаусса в качестве начальных условий."""
    l = 10.
    c = 10
    sigma = 0.5

    def G(x):
        return 1/np.sqrt(2*np.pi*sigma)*np.exp(-0.5*((x-loc)/sigma)**2)

    u_l = 0 if bc_left == 'u=0' else None
    tau = (l/N)/c
    umax = 1.1*G(loc)
    if ic == 'u':
        # u(x,0)=Gaussian, u_t(x,0)=0
        cpu = viz(G, None, None, c, u_l, None, l, tau, gamma, T,
                  umin=-umax, umax=umax, version=version, animate=animate)
    else:
        # u(x,0)=0, u_t(x,0)=Gaussian
        cpu = viz(None, G, None, c, u_l, None, l, tau, gamma, T,
                  umin=-umax/6, umax=umax/6, version=version, animate=animate)

def test_plug():
    """Тестирование возвращается для профиль-площадка после одного периода."""
    l = 1.0
    c = 0.5
    tau = (l/10)/c  # N=10
    I = lambda x: 0 if abs(x-l/2.0) > 0.1 else 1

    u_s, x, t, cpu = solver(
        I=I,
        V=None, f=None, c=0.5, ul=None, ur=None, l=l,
        tau=tau, gamma=1, T=4, user_action=None, version='scalar')
    u_v, x, t, cpu = solver(
        I=I,
        V=None, f=None, c=0.5, ul=None, ur=None, l=l,
        tau=tau, gamma=1, T=4, user_action=None, version='vectorized')
    tol = 1E-13
    diff = abs(u_s - u_v).max()
    assert diff < tol
    u_0 = np.array([I(x_) for x_ in x])
    diff = np.abs(u_s - u_0).max()
    assert diff < tol

def guitar(gamma=1, N=50, animate=True, version='scalar', T=2):
    """Треугольное начальное условияе для моделирования гитарной струны."""
    l = 1.
    c = 1
    x0 = 0.8*l
    tau = l/N/c
    I = lambda x: x/x0 if x < x0 else 1./(1-x0)*(1-x)
    ul = None; ur = None

    cpu = viz(I, None, None, c, ul, ur, l, tau, gamma, T,
              umin=-1.1, umax=1.1, version=version, animate=True)
    print 'CPU time: %s version =' % version, cpu


def moving_end(gamma=1, N=50, reflecting_right_boundary=True, T=2,
               version='vectorized'):
    """
    Изменение по синусоиде y на левой границе.
    Правая граница может быть отражающей или u=0, в соответствии с
    reflecting_right_boundary.
    """
    l = 1.
    c = 1
    tau = l/N/c
    I = lambda x: 0

    def ul(t):
        return (0.25*np.sin(6*np.pi*t) \
                if ((t < 1./6) or \
                    (0.5 + 3./12 <= t <= 0.5 + 4./12 + 0.0001) or \
                    (1.5 <= t <= 1.5 + 1./3 + 0.0001)) \
                else 0)

    if reflecting_right_boundary:
        ur = None
    else:
        ur = 0
    umax = 1.1*0.5
    cpu = viz(I, None, None, c, ul, ur, l, tau, gamma, T,
              umin=-umax, umax=umax, version=version, animate=True)
    print 'CPU time: %s version =' % version, cpu


def sincos(gamma=1):
    """Тестируем точное аналитическое решение 
    (синус по пространству, косинус по времени)."""
    l = 10.0
    c = 1
    T = 5
    N = 80
    tau = (l/N)/c

    def u_exact(x, t):
        m = 3.0
        return np.cos(m*np.pi/l*t)*np.sin(m*np.pi/(2*l)*x)

    I = lambda x: u_exact(x, 0)
    ul = lambda t: u_exact(0, t)
    ur = None # условия Неймана

    cpu = viz(I, None, None, c, ul, ur, l, tau, gamma, T,
              umin=-1.1, umax=1.1, version='scalar', animate=True)

    # Анализ сходимости
    def action(y, x, t, n):
        e = np.abs(y - u_exact(x, t[n])).max()
        errors_in_time.append(e)

    E = []
    tau = []
    N_values = [10, 20, 40, 80, 160]
    for N in N_values:
        errors_in_time = []
        tau = (l/N)/c
        solver(I, None, None, c, ul, ur, l, tau, gamma, T,
               user_action=action, version='scalar')
        E.append(max(errors_in_time))
        _h = l/N
        _tau = gamma*_h/c
        tau.append(_tau)
        print tau[-1], E[-1]
    return tau, E

if __name__ == '__main__':
    test_constant()
    test_quadratic()
    test_plug()
