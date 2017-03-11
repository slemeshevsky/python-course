# -*- coding: utf-8 -*-

from vib_undamped import solver, plot_empirical_freq_and_amplitude
from math import pi

tau_values = [0.1, 0.5, 0.01]
u_cases = []
t_cases = []

for tau in tau_values:
	# Расчитываем безразмерную модель для 40 периодов
	u, t = solver(U = 1, omega = 2*pi, tau = tau, T = 40)
	u_cases.append(u)
	t_cases.append(t)

plot_empirical_freq_and_amplitude(u_cases, t_cases, U = 1, omega = 2*pi)
