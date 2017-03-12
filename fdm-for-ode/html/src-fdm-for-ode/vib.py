# -*- coding: utf-8 -*-

import numpy as np
#import matplotlib.pyplot as plt
import scitools.std as plt

def solver(U, V, m, b, s, F, tau, T, damping='linear'):
    """
    Решает задачу m*u'' + f(u') + s(u) = F(t) for t in (0,T],
    u(0)=U и u'(0)=V,
    конечно-разностной схемой с шагом tau.
    Если затухание 'liniear', то f(u')=b*u, если затухание 'quadratic', 
    то f(u')=b*u'*abs(u').
    F(t) и s(u) --- функции Python.
    """
    tau = float(tau); b = float(b); m = float(m) # avoid integer div.
    N = int(round(T/tau))
    u = np.zeros(N+1)
    t = np.linspace(0, N*tau, N+1)

    u[0] = U
    if damping == 'linear':
        u[1] = u[0] + tau*V + tau**2/(2*m)*(-b*V - s(u[0]) + F(t[0]))
    elif damping == 'quadratic':
        u[1] = u[0] + tau*V + \
               tau**2/(2*m)*(-b*V*abs(V) - s(u[0]) + F(t[0]))

    for n in range(1, N):
        if damping == 'linear':
            u[n+1] = (2*m*u[n] + (b*tau/2 - m)*u[n-1] +
                      tau**2*(F(t[n]) - s(u[n])))/(m + b*tau/2)
        elif damping == 'quadratic':
            u[n+1] = (2*m*u[n] - m*u[n-1] + b*u[n]*abs(u[n] - u[n-1])
                      + tau**2*(F(t[n]) - s(u[n])))/\
                      (m + b*abs(u[n] - u[n-1]))
    return u, t

def visualize(u, t, title='', filename='tmp'):
    plt.plot(t, u, 'b-')
    plt.xlabel('t')
    plt.ylabel('u')
    tau = t[1] - t[0]
    plt.title('tau=%g' % tau)
    umin = 1.2*u.min(); umax = 1.2*u.max()
    plt.axis([t[0], t[-1], umin, umax])
    plt.title(title)
    plt.savefig(filename + '.png')
    plt.savefig(filename + '.pdf')
    plt.show()

import sympy as sym

def test_constant():
    """Тестирование постоянного решения."""
    u_exact = lambda t: U
    U = 1.2; V = 0; m = 2; b = 0.9
    omega = 1.5
    s = lambda u: omega**2*u
    F = lambda t: omega**2*u_exact(t)
    tau = 0.2
    T = 2
    u, t = solver(U, V, m, b, s, F, tau, T, 'linear')
    difference = np.abs(u_exact(t) - u).max()
    tol = 1E-13
    assert difference < tol

    u, t = solver(U, V, m, b, s, F, tau, T, 'quadratic')
    difference = np.abs(u_exact(t) - u).max()
    assert difference < tol

def lhs_eq(t, m, b, s, u, damping='linear'):
    """Возвращает левую часть дифференциального уравнения как выражение sympy."""
    v = sym.diff(u, t)
    if damping == 'linear':
        return m*sym.diff(u, t, t) + b*v + s(u)
    else:
        return m*sym.diff(u, t, t) + b*v*sym.Abs(v) + s(u)

def test_quadratic():
    """Тестирование квадратичного решения."""
    U = 1.2; V = 3; m = 2; b = 0.9
    s = lambda u: 4*u
    t = sym.Symbol('t')
    tau = 0.2
    T = 2

    q = 2  # произвольная постоянная
    u_exact = U + V*t + q*t**2
    F = sym.lambdify(t, lhs_eq(t, m, b, s, u_exact, 'linear'))
    u_exact = sym.lambdify(t, u_exact, modules='numpy')
    u1, t1 = solver(U, V, m, b, s, F, tau, T, 'linear')
    diff = np.abs(u_exact(t1) - u1).max()
    tol = 1E-13
    assert diff < tol

    # В случае квадратичного затухания u_exact должно быть линейным,
    # для того чтобы можно было точно восстановить его
    u_exact = U + V*t
    F = sym.lambdify(t, lhs_eq(t, m, b, s, u_exact, 'quadratic'))
    u_exact = sym.lambdify(t, u_exact, modules='numpy')
    u2, t2 = solver(U, V, m, b, s, F, tau, T, 'quadratic')
    diff = np.abs(u_exact(t2) - u2).max()
    assert diff < tol

def test_sinusoidal():
    """Тестировние численного точного синусоидального решения при b=F=0."""
    from math import asin

    def u_exact(t):
        omega_numerical = 2/tau*np.arcsin(omega*tau/2)
        return U*np.cos(omega_numerical*t)

    U = 1.2; V = 0; m = 2; b = 0
    omega = 1.5  # фиксируем частоту
    s = lambda u: m*omega**2*u
    F = lambda t: 0
    tau = 0.2
    T = 6
    u, t = solver(U, V, m, b, s, F, tau, T, 'linear')
    diff = np.abs(u_exact(t) - u).max()
    tol = 1E-14
    assert diff < tol

    u, t = solver(U, V, m, b, s, F, tau, T, 'quadratic')
    diff = np.abs(u_exact(t) - u).max()
    assert diff < tol

def test_mms():
    """Используем метод пробных решений."""
    m = 4.; b = 1
    omega = 1.5
    t = sym.Symbol('t')
    u_exact = 3*sym.exp(-0.2*t)*sym.cos(1.2*t)
    U = u_exact.subs(t, 0).evalf()
    V = sym.diff(u_exact, t).subs(t, 0).evalf()
    u_exact_py = sym.lambdify(t, u_exact, modules='numpy')
    s = lambda u: u**3
    tau = 0.2
    T = 6
    errors_linear = []
    errors_quadratic = []
    # Выполняем сгущение сетки и вычисляем погрешность
    for i in range(5):
        F_formula = lhs_eq(t, m, b, s, u_exact, 'linear')
        F = sym.lambdify(t, F_formula)
        u1, t1 = solver(U, V, m, b, s, F, tau, T, 'linear')
        error = np.sqrt(np.sum((u_exact_py(t1) - u1)**2)*tau)
        errors_linear.append((tau, error))

        F_formula = lhs_eq(t, m, b, s, u_exact, 'quadratic')
        #print sym.latex(F_formula, mode='plain')
        F = sym.lambdify(t, F_formula)
        u2, t2 = solver(U, V, m, b, s, F, tau, T, 'quadratic')
        error = np.sqrt(np.sum((u_exact_py(t2) - u2)**2)*tau)
        errors_quadratic.append((tau, error))
        tau /= 2
    # Оценка скорости сходимости
    tol = 0.05
    for errors in errors_linear, errors_quadratic:
        for i in range(1, len(errors)):
            tau, error = errors[i]
            tau_1, error_1 = errors[i-1]
            r = np.log(error/error_1)/np.log(tau/tau_1)
            assert abs(r - 2.0) < tol

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--U', type=float, default=1.0)
    parser.add_argument('--V', type=float, default=0.0)
    parser.add_argument('--m', type=float, default=1.0)
    parser.add_argument('--b', type=float, default=0.0)
    parser.add_argument('--s', type=str, default='u')
    parser.add_argument('--F', type=str, default='0')
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--T', type=float, default=10)
    parser.add_argument('--window_width', type=float, default=30.,
                        help='Number of periods in a window')
    parser.add_argument('--damping', type=str, default='linear')
    parser.add_argument('--savefig', action='store_true')
    parser.add_argument('--SCITOOLS_easyviz_backend', default='matplotlib')
    a = parser.parse_args()
    from scitools.std import StringFunction
    s = StringFunction(a.s, independent_variable='u')
    F = StringFunction(a.F, independent_variable='t')
    U, V, m, b, tau, T, window_width, savefig, damping = \
       a.U, a.V, a.m, a.b, a.tau, a.T, a.window_width, a.savefig, \
       a.damping

    u, t = solver(U, V, m, b, s, F, tau, T, damping)
    num_periods = plot_empirical_freq_and_amplitude(u, t)
    num_periods = 4
    tit = 'tau = %g' % tau
    if num_periods <= 40:
        plt.figure()
        visualize(u, t, title=tit)
    else:
        visualize_front(u, t, window_width, savefig)
        visualize_front_ascii(u, t)
    show()

def plot_empirical_freq_and_amplitude(u, t):
    minima, maxima = minmax(t, u)
    p = periods(maxima)
    a = amplitudes(minima, maxima)
    plt.figure()
    from math import pi
    omega = 2*pi/p
    plt.plot(range(len(p)), omega, 'r-')
    plt.hold('on')
    plt.plot(range(len(a)), a, 'b-')
    ymax = 1.1*max(omega.max(), a.max())
    ymin = 0.9*min(omega.min(), a.min())
    plt.axis([0, max(len(p), len(a)), ymin, ymax])
    plt.legend(['estimated frequency', 'estimated amplitude'],
               loc='upper right')
    return len(maxima)

def visualize_front(u, t, window_width, savefig=False):
	"""
	Визуализация приближенного и точного решений с использованием
	moving plot window и непрерывное изменение кривых от времени.
    P - приближенное значение периода.
    """
	import scitools.std as st
	from scitools.MovingPlotWindow import MovingPlotWindow

	umin = 1.2*u.min();  umax = -umin
	plot_manager = MovingPlotWindow(
		window_width=window_width,
		tau=t[1]-t[0],
		yaxis=[umin, umax],
		mode='continuous drawing')
	for n in range(1,len(u)):
		if plot_manager.plot(n):
			s = plot_manager.first_index_in_plot
			st.plot(t[s:n+1], u[s:n+1], 'r-1',
			        title='t=%6.3f' % t[n],
			        axis=plot_manager.axis(),
			        show=not savefig) # drop window if savefig
			if savefig:
				print 't=%g' % t[n]
				st.savefig('tmp_vib%04d.png' % n)
        plot_manager.update(n)

def visualize_front_ascii(u, t, fps=10):
    """
    Визуализация приближенного и точного решений в коне 
    терминала (используются только символы ascii).
    """
    from scitools.avplotter import Plotter
    import time
    umin = 1.2*u.min();  umax = -umin

    p = Plotter(ymin=umin, ymax=umax, width=60, symbols='+o')
    for n in range(len(u)):
        print p.plot(t[n], u[n]), '%.2f' % (t[n])
        time.sleep(1/float(fps))

def minmax(t, u):
    """
    Вычисляем все локальные минимумы и максимумы приближенного
    решения u(t), заданного массивами  u и t.
    Возвращает список минимумов и максимумов вида (t[i],u[i]).
    """
    minima = []; maxima = []
    for n in range(1, len(u)-1, 1):
        if u[n-1] > u[n] < u[n+1]:
            minima.append((t[n], u[n]))
        if u[n-1] < u[n] > u[n+1]:
            maxima.append((t[n], u[n]))
    return minima, maxima

def periods(extrema):
    """
    По заданному списку точек минимума и максимума (t,u),
    возвращает массив соотвествующих локльных периодов.
    """
    p = [extrema[n][0] - extrema[n-1][0]
         for n in range(1, len(extrema))]
    return np.array(p)

def amplitudes(minima, maxima):
    """
    По заданным спискам точек минимума и максимума (t,u), 
    возвращает массив соответствующих локальных амплитуд.
    """
    # Сравниваем первый максимум с первым минимумом и т.д.
    a = [(abs(maxima[n][1] - minima[n][1]))/2.0
         for n in range(min(len(minima),len(maxima)))]
    return np.array(a)

if __name__ == '__main__':
    main()
    #test_constant()
    #test_sinusoidal()
    #test_mms()
    #test_quadratic()
