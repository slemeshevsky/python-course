# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def solver(U, omega, tau, T):
    """
    Решается задача
    u'' + omega**2*u = 0 для t из (0,T], u(0)=U и u'(0)=0,
    конечно-разностным методом с постоянным шагом tau
    """
    tau = float(tau)
    Nt = int(round(T/tau))
    u = np.zeros(Nt+1)
    t = np.linspace(0, Nt*tau, Nt+1)

    u[0] = U
    u[1] = u[0] - 0.5*tau**2*omega**2*u[0]
    for n in range(1, Nt):
        u[n+1] = 2*u[n] - u[n-1] - tau**2*omega**2*u[n]
    return u, t

def u_exact(t, U, omega):
    return U*np.cos(omega*t)

def visualize(u, t, U, omega):
    plt.plot(t, u, 'r--o')
    t_fine = np.linspace(0, t[-1], 1001)  # мелкая сетка для точного решения
    u_e = u_exact(t_fine, U, omega)
    plt.hold('on')
    plt.plot(t_fine, u_e, 'b-')
    plt.legend([u'приближенное', u'точное'], loc='upper left')
    plt.xlabel('$t$')
    plt.ylabel('$u$')
    tau = t[1] - t[0]
    plt.title('$\\tau = $ %g' % tau)
    umin = 1.2*u.min();  umax = -umin
    plt.axis([t[0], t[-1], umin, umax])
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')

def test_three_steps():
    from math import pi
    U = 1;  omega = 2*pi;  tau = 0.1;  T = 1
    u_by_hand = np.array([
	    1.000000000000000,
	    0.802607911978213,
	    0.288358920740053])
    u, t = solver(U, omega, tau, T)
    diff = np.abs(u_by_hand - u[:3]).max()
    tol = 1E-14
    assert diff < tol

def convergence_rates(m, solver_function, num_periods=8):
    """
    Возвращает m-1 эмпирическую оценку скорости сходимости, 
    полученную на основе m расчетов, для каждого из которых 
    шаг по времени уменьшается в два раза.
    solver_function(U, omega, tau, T) решает каждую задачу, 
    для которой T, получается на основе вычислений для 
    num_periods периодов.
    """
    from math import pi
    omega = 0.35; U = 0.3       # просто заданные значения
    P = 2*pi/omega              # период
    tau = P/30                  # 30 шагов на период 2*pi/omega
    T = P*num_periods

    tau_values = []
    E_values = []
    for i in range(m):
        u, t = solver_function(U, omega, tau, T)
        u_e = u_exact(t, U, omega)
        E = np.sqrt(tau*np.sum((u_e-u)**2))
        tau_values.append(tau)
        E_values.append(E)
        tau = tau/2

    r = [np.log(E_values[i-1]/E_values[i])/
         np.log(tau_values[i-1]/tau_values[i])
         for i in range(1, m, 1)]
    return r

def test_convergence_rates():
    r = convergence_rates(m=5, solver_function=solver, num_periods=8)
    tol = 0.1
    assert abs(r[-1] - 2.0) < tol

def main(solver_function=solver):
    import argparse
    from math import pi
    parser = argparse.ArgumentParser()
    parser.add_argument('--U', type=float, default=1.0)
    parser.add_argument('--omega', type=float, default=2*pi)
    parser.add_argument('--tau', type=float, default=0.05)
    parser.add_argument('--num_periods', type=int, default=5)
    parser.add_argument('--savefig', action='store_true')
    # Хак для использования параметров --SCITOOLS (считываются, когда импортируется scitools.std)
    parser.add_argument('--SCITOOLS_easyviz_backend', default='matplotlib')
    a = parser.parse_args()
    U, omega, tau, num_periods, savefig = \
       a.U, a.omega, a.tau, a.num_periods, a.savefig

    P = 2*pi/omega  # один период
    T = P*num_periods
    u, t = solver_function(U, omega, tau, T)
    if num_periods <= 10:
        visualize(u, t, U, omega)
    else:
        visualize_front(u, t, U, omega, savefig)
        #visualize_front_ascii(u, t, U, omega)
    #plot_empirical_freq_and_amplitude(u, t, U, omega)
    plt.show()

def plot_empirical_freq_and_amplitude(u, t, U, omega):
    """
    Находит эмпирически угловую частоту и амплитуду при вычислениях,
    зависящую от u и t. u и t могут быть массивами или (в случае 
    нескольких расчетов) многомерными массивами.
    Одно построение графика выполняется для амплитуды и одно для 
    угловой частоты (на легендах названа просто частотой).
    """
    from vib_empirical_analysis import minmax, periods, amplitudes
    from math import pi
    if not isinstance(u, (list,tuple)):
        u = [u]
        t = [t]
    legends1 = []
    legends2 = []
    for i in range(len(u)):
        minima, maxima = minmax(t[i], u[i])
        p = periods(maxima)
        a = amplitudes(minima, maxima)
        plt.figure(1)
        plt.plot(range(len(p)), 2*pi/p)
        legends1.append(u'Частота, case%d' % (i+1))
        plt.hold('on')
        plt.figure(2)
        plt.plot(range(len(a)), a)
        plt.hold('on')
        legends2.append(u'Амплитуда, case%d' % (i+1))
    plt.figure(1)
    plt.plot(range(len(p)), [omega]*len(p), 'k--')
    legends1.append(u'Точная частота')
    plt.legend(legends1, loc='lower left')
    plt.axis([0, len(a)-1, 0.8*omega, 1.2*omega])
    plt.savefig('tmp1.png');  plt.savefig('tmp1.pdf')
    plt.figure(2)
    plt.plot(range(len(a)), [U]*len(a), 'k--')
    legends2.append(u'Точная амплитуда')
    plt.legend(legends2, loc='lower left')
    plt.axis([0, len(a)-1, 0.8*U, 1.2*U])
    plt.savefig('tmp2.png');  plt.savefig('tmp2.pdf')
    plt.show()


def visualize_front(u, t, U, omega, savefig=False, skip_frames=1):
    """
    Стороится зависимость приближенного и точного решений
    от t с использованием анимированного изображения и непрерывного
    отображения кривых, изменяющихся со временем.
    Графики сохраняются в файлы, если параметр savefig=True.
    Только каждый skip_frames-й график сохраняется (например, если 
    skip_frame=10, только каждый десятый график сохраняется в файл;
    это удобно, если нужно сравнивать графики для различных моментов
    времени).
    """
    import scitools.std as st
    from scitools.MovingPlotWindow import MovingPlotWindow
    from math import pi

    # Удаляем все старые графики tmp_*.png
    import glob, os
    for filename in glob.glob('tmp_*.png'):
        os.remove(filename)

    P = 2*pi/omega  # один период
    umin = 1.2*u.min();  umax = -umin
    tau = t[1] - t[0]
    plot_manager = MovingPlotWindow(
        window_width=8*P,
        dt=tau,
        yaxis=[umin, umax],
        mode='continuous drawing')
    frame_counter = 0
    for n in range(1,len(u)):
        if plot_manager.plot(n):
            s = plot_manager.first_index_in_plot
            st.plot(t[s:n+1], u[s:n+1], 'r-1',
                    t[s:n+1], U*np.cos(omega*t)[s:n+1], 'b-1',
                    title='t=%6.3f' % t[n],
                    axis=plot_manager.axis(),
                    show=not savefig) # пропускаем окно, если savefig
            if savefig and n % skip_frames == 0:
                filename = 'tmp_%04d.png' % frame_counter
                st.savefig(filename)
                print u'Создаем графический файл', filename, 't=%g' % t[n]
                frame_counter += 1
        plot_manager.update(n)

def bokeh_plot(u, t, legends, U, omega, t_range, filename):
	"""
	Строится график зависимости приближенного решения от t с 
	использованием библиотеки Bokeh.
	u и t - списки (несколько экспериментов могут сравниваться).
	легенды содержат строки для различных пар u,t.
    """
	if not isinstance(u, (list,tuple)):
		u = [u]  
	if not isinstance(t, (list,tuple)):
		t = [t]  
	if not isinstance(legends, (list,tuple)):
		legends = [legends] 

	import bokeh.plotting as plt
	plt.output_file(filename, mode='cdn', title=u'Сравнение с помощью Bokeh')
	# Предполагаем, что все массивы t имеют одинаковые размеры
	t_fine = np.linspace(0, t[0][-1], 1001)  # мелкая сетка для точного решения
	tools = 'pan,wheel_zoom,box_zoom,reset,'\
	        'save,box_select,lasso_select'
	u_range = [-1.2*U, 1.2*U]
	font_size = '8pt'
	p = []  # список графических объектов
	# Создаем первую фигуру
	p_ = plt.figure(
		width=300, plot_height=250, title=legends[0],
		x_axis_label='t', y_axis_label='u',
		x_range=t_range, y_range=u_range, tools=tools,
		title_text_font_size=font_size)
	p_.xaxis.axis_label_text_font_size=font_size
	p_.yaxis.axis_label_text_font_size=font_size
	p_.line(t[0], u[0], line_color='blue')
	# Добавляем точное решение
	u_e = u_exact(t_fine, U, omega)
	p_.line(t_fine, u_e, line_color='red', line_dash='4 4')
	p.append(p_)
	# Создаем оставшиеся фигуры и добавляем их оси к осям первой фигуры
	for i in range(1, len(t)):
		p_ = plt.figure(
			width=300, plot_height=250, title=legends[i],
			x_axis_label='t', y_axis_label='u',
			x_range=p[0].x_range, y_range=p[0].y_range, tools=tools,
			title_text_font_size=font_size)
		p_.xaxis.axis_label_text_font_size = font_size
		p_.yaxis.axis_label_text_font_size = font_size
		p_.line(t[i], u[i], line_color='blue')
		p_.line(t_fine, u_e, line_color='red', line_dash='4 4')
		p.append(p_)
		
	# Располагаем все графики на сетке с 3 графиками в строке
	grid = [[]]
	for i, p_ in enumerate(p):
		grid[-1].append(p_)
		if (i+1) % 3 == 0:
			# Новая строка
			grid.append([])
	plot = plt.gridplot(grid, toolbar_location='left')
	plt.save(plot)
	plt.show(plot)

	
def demo_bokeh():
    """Решаем обезразмеренное ОДУ u'' + u = 0."""
    omega = 1.0        # обезразмеренная задача (частота)
    P = 2*np.pi/omega  # период
    num_steps_per_period = [5, 10, 20, 40, 80]
    T = 40*P       # Время моделирования: 40 периодов
    u = []         # список с приближенными решениями
    t = []         # список с соответствующими сетками
    legends = []
    for n in num_steps_per_period:
        tau = P/n
        u_, t_ = solver(U=1, omega=omega, tau=tau, T=T)
        u.append(u_)
        t.append(t_)
        legends.append(u'Шагов на период: %d' % n)
    bokeh_plot(u, t, legends, U=1, omega=omega, t_range=[0, 4*P],
               filename='bokeh.html')

if __name__ == '__main__':
    #main()
    demo_bokeh()
#    raw_input()
