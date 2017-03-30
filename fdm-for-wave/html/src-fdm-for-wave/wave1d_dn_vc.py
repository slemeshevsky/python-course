#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Решение одномерного волнового уравнения с однородными 
граничными условиями Неймана
 y, x, t, cpu = solver(I, V, f, c, ul, ur, l, tau, gamma, T,
                       user_action=None, version='scalar',
                       stability_safety_factor=1.0)

Функция solver решает волновое уравнение 
u_tt = (c**2*u_x)_x + f(x,t) on (0,l) 
c u=ul или du/dn=0 при x=0, и u=ur или du/dn=0
при x = l. Если ul или ur равны None, используется условие du/dn=0, 
иначе используется условие Дирихле  ul(t) и/или ur(t).
Начальные условия: u=I(x), u_t=V(x).

tau --- шаг сетки по времени
T --- конечный момент времени
gamma --- число Куранта (=max(c)*tau/h)
stability_safety_factor вводит критерий устойчивости:
gamma <= stability_safety_factor (<=1).

I, f, ul, ur, и c являются функциями: I(x), f(x,t), ul(t),
ur(t), c(x).
ul и ur могут принимать значения 0 или None, где None соответствует
условию Неймана. f и V также могут принимать значения 0 или None
(эквивалентно 0). c может быть числом или функцией c(x).

user_action --- функция от (y, x, t, n), где вызываемый код может 
добавлять визуализацию, вычисление погрешности, анализ данных, сохранение 
решения и т.д.
"""
import time, glob, shutil, os
import numpy as np

def solver(I, V, f, c, ul, ur, l, tau, gamma, T,
           user_action=None, version='scalar',
           stability_safety_factor=1.0):
    """Решается уравнение $u_tt=(c^2*u_x)_x + f$ на $(0,l)\times(0,T]$."""
    Nt = int(round(T/tau))
    t = np.linspace(0, Nt*tau, Nt+1)      # Сетка по времени

    # Находим max(c) используя мнимую сетку и адаптируем пространственный
    # шаг h согласно gamma и tau
    if isinstance(c, (float,int)):
        c_max = c
    elif callable(c):
        c_max = max([c(x_) for x_ in np.linspace(0, l, 101)])
    h = tau*c_max/(stability_safety_factor*gamma)
    Nx = int(round(l/h))
    x = np.linspace(0, l, Nx+1)          # Пространственная сетка

    # Представляем c(x) как массив
    if isinstance(c, (float,int)):
        c = np.zeros(x.shape) + c
    elif callable(c):
        # Вызываем c(x) и заполняем массив c
        c_ = np.zeros(x.shape)
        for i in range(Nx+1):
            c_[i] = c(x[i])
        c = c_

    q = c**2
    gamma2 = (tau/h)**2; tau2 = tau*tau    # Вспомогательные переменные

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
    if ur is not None:
        if isinstance(ur, (float,int)) and ur == 0:
            ur = lambda t: 0

    # Делаем хэш всех входных данных
    import hashlib, inspect
    data = inspect.getsource(I) + '_' + inspect.getsource(V) + \
           '_' + inspect.getsource(f) + '_' + str(c) + '_' + \
           ('None' if ul is None else inspect.getsource(ul)) + \
           ('None' if ur is None else inspect.getsource(ur)) + \
           '_' + str(l) + str(tau) + '_' + str(gamma) + '_' + str(T) + \
           '_' + str(stability_safety_factor)
    hashed_input = hashlib.sha1(data).hexdigest()
    if os.path.isfile('.' + hashed_input + '_archive.npz'):
        # Расчет уже запущен
        return -1, hashed_input

    y   = np.zeros(Nx+1)   # Массив решения на новом слое
    y_1 = np.zeros(Nx+1)   # Решение на слое n
    y_2 = np.zeros(Nx+1)   # Решение на слое n-1

    import time;  t0 = time.clock()  # Измерение процессорного времени

    Ix = range(0, Nx+1)
    It = range(0, Nt+1)

    # Задаем начальные условие в y_1
    for i in range(0,Nx+1):
        y_1[i] = I(x[i])

    if user_action is not None:
        user_action(y_1, x, t, 0)

    # Специальная формула для первого слоя
    for i in Ix[1:-1]:
        y[i] = y_1[i] + tau*V(x[i]) + \
        0.5*gamma2*(0.5*(q[i] + q[i+1])*(y_1[i+1] - y_1[i]) - \
                0.5*(q[i] + q[i-1])*(y_1[i] - y_1[i-1])) + \
        0.5*tau2*f(x[i], t[0])

    i = Ix[0]
    if ul is None:
        # Установка граничных условий du/dn = 0
        # x=0: i-1 -> i+1 так как y[i-1]=y[i+1]
        # x=l: i+1 -> i-1 так как y[i+1]=y[i-1])
        ip1 = i+1
        im1 = ip1  # i-1 -> i+1
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*gamma2*(0.5*(q[i] + q[ip1])*(y_1[ip1] - y_1[i])  - \
                       0.5*(q[i] + q[im1])*(y_1[i] - y_1[im1])) + \
        0.5*tau2*f(x[i], t[0])
    else:
        y[i] = ul(tau)

    i = Ix[-1]
    if ur is None:
        im1 = i-1
        ip1 = im1  # i+1 -> i-1
        y[i] = y_1[i] + tau*V(x[i]) + \
               0.5*gamma2*(0.5*(q[i] + q[ip1])*(y_1[ip1] - y_1[i])  - \
                       0.5*(q[i] + q[im1])*(y_1[i] - y_1[im1])) + \
        0.5*tau2*f(x[i], t[0])
    else:
        y[i] = ur(tau)

    if user_action is not None:
        user_action(y, x, t, 1)

    # Обновляем данные для следущего слоя
    #y_2[:] = y_1;  y_1[:] = y  # безопасно, но медленнее
    y_2, y_1, y = y_1, y, y_2

    for n in It[1:-1]:
        # Расчет во внутренних узлах
        if version == 'scalar':
            for i in Ix[1:-1]:
                y[i] = - y_2[i] + 2*y_1[i] + \
                    gamma2*(0.5*(q[i] + q[i+1])*(y_1[i+1] - y_1[i])  - \
                        0.5*(q[i] + q[i-1])*(y_1[i] - y_1[i-1])) + \
                tau2*f(x[i], t[n])

        elif version == 'vectorized':
            y[1:-1] = - y_2[1:-1] + 2*y_1[1:-1] + \
            gamma2*(0.5*(q[1:-1] + q[2:])*(y_1[2:] - y_1[1:-1]) -
                0.5*(q[1:-1] + q[:-2])*(y_1[1:-1] - y_1[:-2])) + \
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
                   gamma2*(0.5*(q[i] + q[ip1])*(y_1[ip1] - y_1[i])  - \
                       0.5*(q[i] + q[im1])*(y_1[i] - y_1[im1])) + \
            tau2*f(x[i], t[n])
        else:
            y[i] = ul(t[n+1])

        i = Ix[-1]
        if ur is None:
            im1 = i-1
            ip1 = im1
            y[i] = - y_2[i] + 2*y_1[i] + \
                   gamma2*(0.5*(q[i] + q[ip1])*(y_1[ip1] - y_1[i])  - \
                       0.5*(q[i] + q[im1])*(y_1[i] - y_1[im1])) + \
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
    return cpu_time, hashed_input


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
    l = 2.5
    c = 1.5
    gamma = 0.75
    Nx = 3  # Очень грубая сетка для теста
    tau = gamma*((l/2)/Nx)/c
    T = 18  

    def assert_no_error(y, x, t, n):
        u_e = u_exact(x, t[n])
        diff = np.abs(y - u_e).max()
        tol = 1E-13
        assert diff < tol

    solver(I, V, f, c, ul, ur, l/2, tau, gamma, T,
           user_action=assert_no_error, version='scalar',
           stability_safety_factor=1)
    solver(I, V, f, c, ul, ur, l/2, tau, gamma, T,
           user_action=assert_no_error, version='vectorized',
           stability_safety_factor=1)

def test_plug():
	"""Тестирование возвращается для профиль-площадка после одного периода."""
	l = 1.
	c = 0.5
	tau = (l/10)/c  # Nx=10
	I = lambda x: 0 if abs(x-l/2.0) > 0.1 else 1

	class Action:
		"""Сохраняем последнее решение."""
        def __call__(self, y, x, t, n):
            if n == len(t)-1:
                self.y = y.copy()
                self.x = x.copy()
                self.t = t[n]


	action = Action()

	solver(
	    I=I,
	    V=None, f=None, c=c, ul=None, ur=None, l=l,
	    tau=tau, gamma=1, T=4, user_action=action, version='scalar')
	u_s = action.y
	solver(
		I=I,
		V=None, f=None, c=c, ul=None, ur=None, l=l,
		tau=tau, gamma=1, T=4, user_action=action, version='vectorized')
	u_v = action.y
	diff = np.abs(u_s - u_v).max()
	tol = 1E-13
	assert diff < tol
	u_0 = np.array([I(x_) for x_ in action.x])
	diff = np.abs(u_s - u_0).max()
	assert diff < tol

def merge_zip_archives(individual_archives, archive_name):
	"""
	Слияние индивидуальных zip-архивов, сделанных с помощью
	numpy.savez, в один архив с именем archive_name.
	Отдельные архивы могут быть заданы как список имен.
	В результате выполнения этой функции все отдельные 
	архивы удаляются и создается один новый архив.
	"""
	import zipfile
	archive = zipfile.ZipFile(
		archive_name, 'w', zipfile.ZIP_DEFLATED,
		allowZip64=True)
	if isinstance(individual_archives, (list,tuple)):
		filenames = individual_archives
	elif isinstance(individual_archives, str):
		filenames = glob.glob(individual_archives)

	# Открываем каждый архив и пишем его в общий архив
	for filename in filenames:
		f = zipfile.ZipFile(filename,  'r',
		                    zipfile.ZIP_DEFLATED)
		for name in f.namelist():
			data = f.open(name, 'r')
			# Сохраняем под именем без .npy
			archive.writestr(name[:-4], data.read())
		f.close()
		os.remove(filename)
	archive.close()

class PlotAndStoreSolution:
	"""
	Класс для функиции user_action в solver.
	Только визуализация решения.
	"""
	def __init__(
			self,
			casename='tmp',    # Префикс в именах файлов
			umin=-1, umax=1,   # Задаются границы по оси y
			pause_between_frames=None,  # Скорость видео
			backend='matplotlib',       # или 'gnuplot' или None
			screen_movie=True, # Показывать видео на экране?
			title='',          # Дополнительное сообщение в title
			skip_frame=1,      # Пропуск каждого skip_frame кадра
			filename=None):    # Имя файла с решением
		self.casename = casename
		self.yaxis = [umin, umax]
		self.pause = pause_between_frames
		self.backend = backend
		if backend is None:
			# Использовать matplotlib
			import matplotlib.pyplot as plt
		elif backend in ('matplotlib', 'gnuplot'):
			module = 'scitools.easyviz.' + backend + '_'
			exec('import %s as plt' % module)
		self.plt = plt
		self.screen_movie = screen_movie
		self.title = title
		self.skip_frame = skip_frame
		self.filename = filename
		if filename is not None:
			# Сохранение временной сетки, когда y записывается в файл
			self.t = []
			filenames = glob.glob('.' + self.filename + '*.dat.npz')
			for filename in filenames:
				os.remove(filename)

		# Очистка старых кадров
		for filename in glob.glob('frame_*.png'):
			os.remove(filename)

	def __call__(self, u, x, t, n):
		"""
		Функция обратного вызова user_action, вызываемая солвером:
		сохранение решения, построение графиков на экране и
		и сохранение их в файл.
		"""
		# Сохраняем решение u в файл, используя numpy.savez
		if self.filename is not None:
			name = 'u%04d' % n  # имя массива
			kwargs = {name: u}
			fname = '.' + self.filename + '_' + name + '.dat'
			np.savez(fname, **kwargs)
			self.t.append(t[n])  # сохранение соответствующего временного знаяения
			if n == 0:           # сохранение массива x один раз
				np.savez('.' + self.filename + '_x.dat', x=x)

		# Анимация
		if n % self.skip_frame != 0:
			return
		title = 't=%.3f' % t[n]
		if self.title:
			title = self.title + ' ' + title
		if self.backend is None:
			# анимация matplotlib 
			if n == 0:
				self.plt.ion()
				self.lines = self.plt.plot(x, u, 'r-')
				self.plt.axis([x[0], x[-1],
				               self.yaxis[0], self.yaxis[1]])
				self.plt.xlabel('x')
				self.plt.ylabel('u')
				self.plt.title(title)
				self.plt.legend(['t=%.3f' % t[n]])
			else:
				# Обновляем решение
				self.lines[0].set_ydata(u)
				self.plt.legend(['t=%.3f' % t[n]])
				self.plt.draw()
		else:
			# анимация scitools.easyviz 
			self.plt.plot(x, u, 'r-',
			              xlabel='x', ylabel='u',
			              axis=[x[0], x[-1],
			                    self.yaxis[0], self.yaxis[1]],
			              title=title,
			              show=self.screen_movie)
		# пауза
		if t[n] == 0:
			time.sleep(2)  # показываем начальное решение 2 с
		else:
			if self.pause is None:
				pause = 0.2 if u.size < 100 else 0
			time.sleep(pause)

		self.plt.savefig('frame_%04d.png' % (n))

	def make_movie_file(self):
		"""
		Создается подкаталог на основе casename, перемещаем все файлы
		с кадрами в этот каталог и создаем файл index.html для показа
		видео в браузере (как последовательности PNG файлов).
		"""
		directory = self.casename
		if os.path.isdir(directory):
			shutil.rmtree(directory)   # rm -rf directory
		os.mkdir(directory)            # mkdir directory
		# mv frame_*.png directory
		for filename in glob.glob('frame_*.png'):
			os.rename(filename, os.path.join(directory, filename))
		os.chdir(directory)        # cd directory
		fps = 4 # frames per second
		if self.backend is not None:
			from scitools.std import movie
			movie('frame_*.png', encoder='html',
			      output_file='index.html', fps=fps)

		# Создаем другие видео форматы: Flash, Webm, Ogg, MP4
		codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
		                 libtheora='ogg')
		filespec = 'frame_%04d.png'
		movie_program = 'avconv' # или 'ffmpeg' 
		for codec in codec2ext:
			ext = codec2ext[codec]
			cmd = '%(movie_program)s -r %(fps)d -i %(filespec)s '\
			      '-vcodec %(codec)s movie.%(ext)s' % vars()
			os.system(cmd)
		os.chdir(os.pardir)  # возвращаемся в родительский каталог

	def close_file(self, hashed_input):
		"""
		Сливаем все файлы в один архив.
		hashed_input --- строка, отражающая входные данные
		для моделирования (создана функцией solver).
		"""
		if self.filename is not None:
			np.savez('.' + self.filename + '_t.dat',
			         t=np.array(self.t, dtype=float))

			archive_name = '.' + hashed_input + '_archive.npz'
			filenames = glob.glob('.' + self.filename + '*.dat.npz')
			merge_zip_archives(filenames, archive_name)
		print 'Archive name:', archive_name
		# data = numpy.load(archive); data.files holds names
		# data[name] extract the array

def demo_BC_plug(gamma=1, Nx=40, T=4):
	"""Граничные условия u=0 и u_x=0 с профилем-площадкой."""
	action = PlotAndStoreSolution(
		'plug', -1.3, 1.3, skip_frame=1,
		title='$u(0,t)=0$, $du(l,t)/dn=0.$', filename='tmpdata')
	# Безразмерная задача: l=1, c=1, max I=1
	l = 1.
	tau = (l/Nx)/gamma  # выбор ограничения устойчивости по заданному Nx
	cpu, hashed_input = solver(
		I=lambda x: 0 if abs(x-l/2.0) > 0.1 else 1,
		V=0, f=0, c=1, ul=lambda t: 0, ur=None, l=l,
		tau=tau, gamma=gamma, T=T,
		user_action=action, version='vectorized',
		stability_safety_factor=1)
	action.make_movie_file()
	if cpu > 0:  
		action.close_file(hashed_input)
	print 'cpu:', cpu

def demo_BC_gaussian(gamma=1, Nx=80, T=4):
    """Граничные условия u=0 и u_x=0 с функцией-шапочкой."""
    # Безразмерная задача: l=1, c=1, max I=1
    action = PlotAndStoreSolution(
        'gaussian', -1.3, 1.3, skip_frame=1,
        title='u(0,t)=0, du(l,t)/dn=0.', filename='tmpdata')
    l = 1.
    tau = (l/Nx)/c # выбор ограничения устойчивости по заданному Nx
    cpu, hashed_input = solver(
        I=lambda x: np.exp(-0.5*((x-0.5)/0.05)**2),
        V=0, f=0, c=1, ul=lambda t: 0, ur=None, l=l,
        tau=tau, gamma=gamma, T=T,
        user_action=action, version='vectorized',
        stability_safety_factor=1)
    action.make_movie_file()
    if cpu > 0: 
        action.close_file(hashed_input)

def moving_end(gamma=1, Nx=50, reflecting_right_boundary=True,
               version='vectorized'):
    # Безразмерная задача: l=1, c=1, max I=1
    l = 1.
    c = 1
    tau = (l/Nx)/c # выбор ограничения устойчивости по заданному Nx
    T = 3
    I = lambda x: 0
    V = 0
    f = 0

    def ul(t):
        return 1.0*sin(6*np.pi*t) if t < 1./3 else 0

    if reflecting_right_boundary:
        ur = None
        bc_right = 'du(l,t)/h=0'
    else:
        ur = 0
        bc_right = 'u(l,t)=0'

    action = PlotAndStoreSolution(
        'moving_end', -2.3, 2.3, skip_frame=4,
        title='u(0,t)=0.25*sin(6*pi*t) if t < 1/3 else 0, '
        + bc_right, filename='tmpdata')
    cpu, hashed_input = solver(
        I, V, f, c, ul, ur, l, tau, gamma, T,
        user_action=action, version=version,
        stability_safety_factor=1)
    action.make_movie_file()
    if cpu > 0: 
        action.close_file(hashed_input)


class PlotMediumAndSolution(PlotAndStoreSolution):
    def __init__(self, medium, **kwargs):
        """Отмечаем среду на графике: medium=[x_l, x_R]."""
        self.medium = medium
        PlotAndStoreSolution.__init__(self, **kwargs)

    def __call__(self, u, x, t, n):
        # Сохраняем решение u в файл, используя numpy.savez
        if self.filename is not None:
            name = 'u%04d' % n 
            kwargs = {name: u}
            fname = '.' + self.filename + '_' + name + '.dat'
            np.savez(fname, **kwargs)
            self.t.append(t[n])
            if n == 0:         
                np.savez('.' + self.filename + '_x.dat', x=x)

        # Анимация
        if n % self.skip_frame != 0:
            return
        # Рисуем u и отмечаем среду x=x_L и x=x_R
        x_L, x_R = self.medium
        umin, umax = self.yaxis
        title = '$N_x$=%d' % (x.size-1)
        if self.title:
            title = self.title + ' ' + title
        if self.backend is None:
            # анимация matplotlib
            if n == 0:
                self.plt.ion()
                self.lines = self.plt.plot(
                    x, u, 'r-',
                    [x_L, x_L], [umin, umax], 'k--',
                    [x_R, x_R], [umin, umax], 'k--')
                self.plt.axis([x[0], x[-1],
                               self.yaxis[0], self.yaxis[1]])
                self.plt.xlabel('$x$')
                self.plt.ylabel('$u$')
                self.plt.title(title)
                self.plt.text(0.75, 1.0, '$\gamma=0.25$')
                self.plt.text(0.32, 1.0, '$\gamma=1$')
                self.plt.legend(['$t=$%.3f' % t[n]])
            else:
                self.lines[0].set_ydata(u)
                self.plt.legend(['$t=$%.3f' % t[n]])
                self.plt.draw()
        else:
            # анимация scitools.easyviz
            self.plt.plot(x, u, 'r-',
                          [x_L, x_L], [umin, umax], 'k--',
                          [x_R, x_R], [umin, umax], 'k--',
                          xlabel='x', ylabel='u',
                          axis=[x[0], x[-1],
                                self.yaxis[0], self.yaxis[1]],
                          title=title,
                          show=self.screen_movie)
        # пауза
        if t[n] == 0:
            time.sleep(2)
        else:
            if self.pause is None:
                pause = 0.2 if u.size < 100 else 0
            time.sleep(pause)

        self.plt.savefig('frame_%04d.png' % (n))

def animate_multiple_solutions(*archives):
    a = [load(archive) for archive in archives]
    # Предполагаем, что имена массивов одинаковы во всех архивах
    raise NotImplementedError  # TODO ...

# --- Start pulse
def pulse(gamma=1,            # максимальное число Куранта
          Nx=200,         # число узлов по пространству
          animate=True,
          version='vectorized',
          T=2,            # конечное время
          loc='left',     # размещение начального условия
          pulse_tp='gaussian',  # pulse/init.cond. 
          slowness_factor=2, # скорость распространения волны в правой среде
          medium=[0.7, 0.9], # отрезок правой области (среды)
          skip_frame=1,      
          sigma=0.05):
	"""
	Различные пико-образные начальные условия на [0,1].
	Скорость распространения волны уменьшается в slowness_factor раз
	венутри среды. Параметр loc может принимать значения 'center' или 'left',
	в зависимости от того, где располагается пик начальных условий.
	Параметр sigma определяет ширину импульса.
	"""
	# Используем безразмерные параметры: l=1 для длины области,
	# c_0=1 для скорости распространения волны вне области.
	l = 1.0
	c_0 = 1.0
	if loc == 'center':
		xc = l/2
	elif loc == 'left':
		xc = 0

	if pulse_tp in ('gaussian','Gaussian'):
		def I(x):
			return np.exp(-0.5*((x-xc)/sigma)**2)
	elif pulse_tp == 'plug':
		def I(x):
			return 0 if abs(x-xc) > sigma else 1
	elif pulse_tp == 'cosinehat':
		def I(x):
			# Один период косинуса
			w = 2
			a = w*sigma
			return 0.5*(1 + np.cos(np.pi*(x-xc)/a)) \
				if xc - a <= x <= xc + a else 0

	elif pulse_tp == 'half-cosinehat':
		def I(x):
			# Половина периода косинуса
			w = 4
			a = w*sigma
			return np.cos(np.pi*(x-xc)/a) \
				if xc - 0.5*a <= x <= xc + 0.5*a else 0
	else:
		raise ValueError(u'Ошибочный_tp="%s"' % pulse_tp)

	def c(x):
		return c_0/slowness_factor \
			if medium[0] <= x <= medium[1] else c_0

	umin=-0.5; umax=1.5*I(xc)
	casename = '%s_Nx%s_sf%s' % \
	           (pulse_tp, Nx, slowness_factor)
	action = PlotMediumAndSolution(
		medium, casename=casename, umin=umin, umax=umax,
		skip_frame=skip_frame, screen_movie=animate,
		backend='matplotlib', filename='tmpdata')

	# Выбор ограничения устойчивости при заданном Nx, худший случай c
	# (меньший gamma будет использовать этот шаг tau, но меньшее Nx)
	tau = (l/Nx)/c_0
	cpu, hashed_input = solver(I=I, V=None, f=None, c=c, ul=None, ur=None,
	                           l=l, tau=tau, gamma=gamma, T=T,
	                           user_action=action, version=version,
	                           stability_safety_factor=1)
	action.make_movie_file()
	action.close_file(hashed_input)
# --- End pulse
	
if __name__ == '__main__':
    pass
