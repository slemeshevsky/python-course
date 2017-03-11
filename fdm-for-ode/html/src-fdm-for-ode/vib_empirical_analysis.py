# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import scitools.std as plt

def minmax(t, u):
    """
    Вычисляются все локальные минимумы и максимумы сеточной функции 
    u(t_n), представленной массивами u и t. Возвращается список минимумов
    и максимумов вида (t[i],u[i]).
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
    По заданному списку (t,u) точек минимума или максимума возвращается
    массив соотвествующих локальных периодов.
    """
    p = [extrema[n][0] - extrema[n-1][0]
         for n in range(1, len(extrema))]
    return np.array(p)

def amplitudes(minima, maxima):
    """
    По заданным спискам точек локальных минимумов и максимумов
    возвращается массив соответсвующих локальных амплитуд.
    """
    # Сравнивается первый максимум с первым минимумом и т.д.
    a = [(abs(maxima[n][1] - minima[n][1]))/2.0
         for n in range(min(len(minima),len(maxima)))]
    return np.array(a)

def test_empirical_analysis():
    t = np.linspace(0, 6*np.pi, 1181)

    u = np.exp(-(t-3*np.pi)**2/12.0)*np.cos(np.pi*(t + 0.6*np.sin(0.25*np.pi*t)))
    plt.plot(t, u, label='signal')
    plt.hold('on')
    minima, maxima = minmax(t, u)
    t_min = [ti for ti, ui in minima]
    t_max = [ti for ti, ui in maxima]
    u_min = [ui for ui, ui in minima]
    u_max = [ui for ui, ui in maxima]
    plt.plot(t_min, u_min, 'bo', label='minima')
    plt.plot(t_max, u_max, 'ro', label='maxima')
    plt.legend()

    plt.figure()
    p = periods(maxima)
    a = amplitudes(minima, maxima)
    plt.plot(range(len(p)), p, 'g--', label='periods')
    plt.hold('on')
    plt.plot(range(len(a)), a, 'y-', label='amplitudes')
    plt.legend()

    p_ref = np.array([
        1.48560059,  2.73158819,  2.30028479,  1.42170379,  1.45365219,
        2.39612999,  2.63574299,  1.45365219,  1.42170379])
    a_ref = np.array([
        0.00123696,  0.01207413,  0.19769443,  0.59800044,  0.90044961,
        0.96007725,  0.42076411,  0.08626735,  0.0203696 ,  0.00312785])
    p_diff = np.abs(p - p_ref).max()
    a_diff = np.abs(a - a_ref).max()
    tol = 1E-7
    assert p_diff < tol
    assert a_diff < tol

if __name__ == '__main__':
    test_empirical_analysis()
    plt.show()
