# -*- coding: utf-8 -*-

import numpy as np

def solver(I, V, f, c, l, tau, gamma, T, user_action=None):
    K = int(round(T/tau))
    t = np.linspace(0, K*tau, K+1)   # Сетка по времени
    dx = tau*c/float(gamma)
    N = int(round(l/dx))
    x = np.linspace(0, l, N+1)       # Пространственная сетка
    C2 = gamma**2                    # вспомогательная переменная
    if f is None or f == 0 :
        f = lambda x, t: 0
    if V is None or V == 0:
        V = lambda x: 0

    y   = np.zeros(N+1)   # Массив с решением на новом временном слое n+1
    y_1 = np.zeros(N+1)   # Решение на предыдущем слое n
    y_2 = np.zeros(N+1)   # Решение на слое n-1

    import time;  t0 = time.clock()  # для измерения процессорного времени

    # Задаем начальное условие
    for i in range(0,N+1):
        y_1[i] = I(x[i])

    if user_action is not None:
        user_action(y_1, x, t, 0)

    # Используем специальную формулу для расчета на первом
    # временном шаге с учетом du/dt = 0
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
	    for i in range(1, N):
		    y[i] = - y_2[i] + 2*y_1[i] + C2*(y_1[i-1] - 2*y_1[i] + y_1[i+1]) + tau**2*f(x[i], t[n])

	    y[0] = 0; y[N] = 0 # Задаем граничные условия
	    if user_action is not None:
		    if user_action(y, x, t, n+1):
			    break
		# Изменяем переменные перед переходом на следующий
        # временной слой
	    y_2[:] = y_1;  y_1[:] = y

    cpu_time = t0 - time.clock()
    return y, x, t, cpu_time

def test_quadratic():
    """
    Проверяет воспроизводится ли точно решение u(x,t)=x(l-x)(1+t/2).
    """

    def u_exact(x, t):
        return x*(l-x)*(1 + 0.5*t)

    def I(x):
        return u_exact(x, 0)

    def V(x):
        return 0.5*u_exact(x, 0)

    def f(x, t):
        return 2*(1 + 0.5*t)*c**2

    l = 2.5
    c = 1.5
    gamma = 0.75
    N = 6  # Используем грубую сетку
    tau = gamma*(l/N)/c
    T = 18

    def assert_no_error(u, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(u - u_e).max()
        tol = 1E-13
        assert diff < tol

    solver(I, V, f, c, l, tau, gamma, T,
           user_action=assert_no_error)

def viz(
    I, V, f, c, l, tau, gamma, T,  # Параметры задачи
    umin, umax,               # Интервал для отображения u
    animate=True,             # Расчет с анимацией?
    tool='matplotlib',        # 'matplotlib' или 'scitools'
    solver_function=solver,   # Функция, реализующая алгоритм расчета
    ):
    """Запуск солвера и визуализации u на каждом временном слое."""

    def plot_u_st(u, x, t, n):
        """Функция user_action для солвера."""
        plt.plot(x, u, 'r-',
                 xlabel='x', ylabel='u',
                 axis=[0, l, umin, umax],
                 title='t=%f' % t[n], show=True)
        # Начальные данные отображаем на экране в течение 2 сек.
        # Далее меду временными слоями пауза 0.2 сек.
        time.sleep(2) if t[n] == 0 else time.sleep(0.2)
        plt.savefig('frame_%04d.png' % n)  # для генерации видео

    class PlotMatplotlib:
        def __call__(self, u, x, t, n):
            """Функция user_action для солвера."""
            if n == 0:
                plt.ion()
                self.lines = plt.plot(x, u, 'r-')
                plt.xlabel('x');  plt.ylabel('u')
                plt.axis([0, l, umin, umax])
                plt.legend(['t=%f' % t[n]], loc='lower left')
            else:
                self.lines[0].set_ydata(u)
                plt.legend(['t=%f' % t[n]], loc='lower left')
                plt.draw()
            time.sleep(2) if t[n] == 0 else time.sleep(0.2)
            plt.savefig('tmp_%04d.png' % n)  # для генерации видео

    if tool == 'matplotlib':
        import matplotlib.pyplot as plt
        plot_u = PlotMatplotlib()
    elif tool == 'scitools':
        import scitools.std as plt  # scitools.easyviz 
        plot_u = plot_u_st
    import time, glob, os

    # Удаляем старые кадры
    for filename in glob.glob('tmp_*.png'):
        os.remove(filename)

    # Вызываем солвер и выполняем расчет
    user_action = plot_u if animate else None
    u, x, t, cpu = solver_function(
        I, V, f, c, l, tau, gamma, T, user_action)

    # Генерируем видео файлы
    fps = 4  # Количество кадров в секунду
    codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
                     libtheora='ogg')  # Видео форматы
    filespec = 'tmp_%04d.png'
    movie_program = 'ffmpeg'  # или 'avconv'
    for codec in codec2ext:
        ext = codec2ext[codec]
        cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
              '-vcodec %(codec)s movie.%(ext)s' % vars()
        os.system(cmd)

    if tool == 'scitools':
        # Создаем HTML для показа анимации в браузере
        plt.movie('tmp_*.png', encoder='html', fps=fps,
                  output_file='movie.html')
    return cpu

def guitar(gamma):
    """Треугольная волна."""
    l = 0.75
    x0 = 0.8*l
    a = 0.005
    freq = 440
    wavelength = 2*l
    c = freq*wavelength
    omega = 2*np.pi*freq
    num_periods = 1
    T = 2*np.pi/omega*num_periods
    # Выбираем tau таким же, как при условии устойчивости для N=50
    tau = l/50./c

    def I(x):
        return a*x/x0 if x < x0 else a/(l-x0)*(l-x)

    umin = -1.2*a;  umax = -umin
    cpu = viz(I, 0, 0, c, l, tau, gamma, T, umin, umax,
              animate=True, tool='scitools')

if __name__ == '__main__':
    test_quadratic()
    import sys
    try:
        gamma = float(sys.argv[1])
        print u'Число Куранта gamma=%g' % gamma
    except IndexError:
        gamma = 0.85
    print u'Число Куранта: %.2f' % gamma
    guitar(gamma)
