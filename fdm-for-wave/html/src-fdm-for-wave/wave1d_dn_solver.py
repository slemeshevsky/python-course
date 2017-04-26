# -*- coding: utf-8 -*-
from parameters import Parameters
import numpy as np
import glob, time, hashlib, inspect, os, shutil

class WaveProblem1D(Parameters):
	"""
	Входные параметры смешанной задачи для волнового уравнения.
	"""
	def __init__(self):
		self.prm = dict(L=[0.,1.], I=1.0, V=0.0, f=0.0, c=1.,
		                ul=None, ur=None, u_exact=None, T=1.0)
		self.type = dict(L=list, I=(float,callable), V=(float,callable),
		                 f=(float,callable), c=(float,callable),
		                 ul=(float,callable), ur=(float,callable),
		                 u_exact=(float,callable), T=float)
		self.help = dict(L='Прямоугольная расчетная область',
		                 I='Начальная траектроия',
		                 V='Начальная скорость', f='Внешние силы',
		                 c='Скорость распространения волны',
		                 ul='Левое граничное условие',
		                 ur='Правое граниченое условие',
		                 u_exact='Точное решение задачи',
		                 T='Конечный момент времени')
		self._update()
		
	def _update(self):
		if self.prm['f'] is None:
			self.prm['f'] = (lambda x, t: np.zeros_like(x))
		elif isinstance(self.prm['f'],float):
			val = self.prm['f']
			self.prm['f'] = (lambda x, t: np.zeros_like(x) + val)
		if self.prm['I'] is None:
			self.prm['I'] = lambda x: np.zeros_like(x)
		elif isinstance(self.prm['I'],float):
			val = self.prm['I']
			self.prm['I'] = (lambda x: np.zeros_like(x) + val)
		if self.prm['V'] is None:
			self.prm['V'] = (lambda x: np.zeros_like(x))
		elif isinstance(self.prm['V'],float):
			val = self.prm['V']
			self.prm['V'] = (lambda x: np.zeros_like(x) + val)
		if self.prm['ul'] is not None:
			if isinstance(self.prm['ul'], (float,int)):
				val = self.prm['ul']
				self.prm['ul'] = (lambda t: np.zeros_like(t) + val)
		if self.prm['ur'] is not None:
			if isinstance(self.prm['ur'], (float,int)):
				val = self.prm['ul']
				self.prm['ur'] = (lambda t: np.zeros_like(t) + val)

from UniformFDMesh import Mesh, Function

class WaveExplicitSolver1D(Parameters):
	"""
	Численные параметры и алгоритм решения задачи для волнового уравнения.
	"""
	def __init__(self, problem):
		self.problem = problem
		self.prm = dict(dt=0.1, gamma=1., user_action=None,
		                stability_safety_factor=1.0)
		self.type = dict(dt=float, gamma=float, user_action=callable,
		                 stability_safety_factor=float)
		self.help = dict(dt='Шаг сетки по времени', gamma='Число Куранта',
						 user_action='Функция действий пользователя',
						 stability_safety_factor='Коэффициент, гарантирующий устойчивость')
		
	
	def solve(self):
		"""
		Реализация алгоритма по явной разностной схеме.
		"""

		Nt = int(round(self.problem.prm['T']/self.prm['dt']))
		#Находим max(c), используя мнимую сетку и 
		if isinstance(self.problem.prm['c'], (float,int)):
			c_max = self.problem.prm['c']
		elif callable(self.problem.prm['c']):
			c_max = max([self.problem.prm['c'](x_) for x_ in np.linspace(0,self.problem.prm['L'][1],101)])

		# Вычисляем шаг по пространству и генерируем пространственно-временную сетку
		h = self.prm['dt']*c_max/(self.prm['stability_safety_factor']*self.prm['gamma'])
		self.mesh = Mesh(L=self.problem.prm['L'], T=self.problem.prm['T'], Nt=Nt, d=h)

		# Определяем c(x) как сеточную функцию
		c = Function(self.mesh)
		if isinstance(self.problem.prm['c'], float):
			c.u += self.problem.prm['c']
		elif callable(self.problem.prm['c']):
			c.u = self.problem.prm['c'](self.mesh.x[0])

		q = c.u**2
		self.q = c.u**2
		gamma2 = (self.prm['dt']/h)**2; dt2 = self.prm['dt']*self.prm['dt']

		# Делаем хэш всех входных данных
		data = str(c) + '_' + \
		       ('None' if self.problem.prm['ul'] is None else inspect.getsource(self.problem.prm['ul'])) + \
		       ('None' if self.problem.prm['ur'] is None else inspect.getsource(self.problem.prm['ur'])) + \
		       '_' + str(self.problem.prm['L']) + str(self.prm['dt']) + '_' +\
		       str(self.prm['gamma']) + '_' + str(self.problem.prm['T']) + \
		       '_' + str(self.prm['stability_safety_factor'])
		hashed_input = hashlib.sha1(data).hexdigest()
		if os.path.isfile('.' + hashed_input + '_archive.npz'):
			# Расчет уже запущен
			return -1, hashed_input

		# Определяем решения как сеточные функции
		y = Function(self.mesh)       # Решение на новом слое n+1
		y_1 = Function(self.mesh)     # Решение на слое n
		y_2 = Function(self.mesh)     # Решение на слое n-1

		t0 = time.clock() # Измерение процессорного времени

		y_1.u = self.problem.prm['I'](y_1.mesh.x[0]) 

		if self.prm['user_action'] is not None:
			self.prm['user_action'](y_1.u, y_1.mesh.x[0], y_1.mesh.t, 0)

		Ix = range(0, self.mesh.N[0]+1)
		It = range(0, self.mesh.Nt+1)
		
		# Специальная формула для первого слоя
		for i in Ix[1:-1]:
			y.u[i] = y_1.u[i] + self.prm['dt'] * self.problem.prm['V'](self.mesh.x[0][i]) +\
			         0.5*gamma2*(0.5*(q[i]+q[i+1])*(y_1.u[i+1] - y_1.u[i]) -\
			         0.5*(q[i]+q[i-1])*(y_1.u[i] - y_1.u[i-1])) + \
			         0.5*dt2*self.problem.prm['f'](self.mesh.x[0][i], self.mesh.t[0])
		i = Ix[0]
		if self.problem.prm['ul'] is None:
			ip1 = i+1
			im1 = ip1
			y.u[i] = y_1.u[i] + self.prm['dt']*self.problem.prm['V'](self.mesh.x[0][i]) + \
			         0.5*gamma2*(0.5*(q[i]+q[ip1])*(y_1.u[ip1] - y_1.u[i]) -\
			         0.5*(q[i]+q[im1])*(y_1.u[i] - y_1.u[im1])) + \
			         0.5*dt2*self.problem.prm['f'](self.mesh.x[0][i], self.mesh.t[0])
		else:
			y.u[i] = self.problem.prm['ul'](self.prm['dt'])


		i = Ix[-1]
		if self.problem.prm['ur'] is None:
			im1 = i-1
			ip1 = im1
			y.u[i] = y_1.u[i] + self.prm['dt']*self.problem.prm['V'](self.mesh.x[0][i]) + \
			         0.5*gamma2*(0.5*(q[i]+q[ip1])*(y_1.u[ip1] - y_1.u[i]) -\
			         0.5*(q[i]+q[im1])*(y_1.u[i] - y_1.u[im1])) + \
			         0.5*dt2*self.problem.prm['f'](self.mesh.x[0][i], self.mesh.t[0])
		else:
			y.u[i] = self.problem.prm['ur'](self.prm['dt'])

		if self.prm['user_action'] is not None:
			self.prm['user_action'](y.u, y.mesh.x[0], y.mesh.t, 1)

		# Обновляем данные для следующего слоя
		y_2.u, y_1.u, y.u = y_1.u, y.u, y_2.u

		for n in It[1:-1]:
			# Расчет во внутренних узлах
			y.u[1:-1] = - y_2.u[1:-1] + 2*y_1.u[1:-1] + \
			gamma2*(0.5*(q[1:-1]+q[2:])*(y_1.u[2:] - y_1.u[1:-1]) - \
			        0.5*(q[1:-1]+q[:-2])*(y_1.u[1:-1] - y_1.u[:-2])) + \
			+ dt2*self.problem.prm['f'](self.mesh.x[0][1:-1], self.mesh.t[n])

			# Добавляем граничные условия
			i = Ix[0]
			if self.problem.prm['ul'] is None:
				ip1 = i+1
				im1 = ip1
				y.u[i] = y_1.u[i] + self.prm['dt']*self.problem.prm['V'](self.mesh.x[0][i]) + \
				0.5*gamma2*(0.5*(q[i]+q[ip1])*(y_1.u[ip1] - y_1.u[i]) -\
				            0.5*(q[i]+q[im1])*(y_1.u[i] - y_1.u[im1])) + \
				0.5*dt2*self.problem.prm['f'](self.mesh.x[0][i], self.mesh.t[n])
			else:
				y.u[i] = self.problem.prm['ul'](self.mesh.t[n+1])


			i = Ix[-1]
			if self.problem.prm['ur'] is None:
				im1 = i-1
				ip1 = im1
				y.u[i] = y_1.u[i] + self.prm['dt']*self.problem.prm['V'](self.mesh.x[0][i]) + \
				0.5*gamma2*(0.5*(q[i]+q[ip1])*(y_1.u[ip1] - y_1.u[i]) -\
				            0.5*(q[i]+q[im1])*(y_1.u[i] - y_1.u[im1])) + \
				0.5*dt2*self.problem.prm['f'](self.mesh.x[0][i], self.mesh.t[n])
			else:
				y.u[i] = self.problem.prm['ur'](self.mesh.t[n+1])					

			if self.prm['user_action'] is not None:
				if self.prm['user_action'](y.u, y.mesh.x[0], y.mesh.t, n+1):
					break

			y_2.u, y_1.u, y.u = y_1.u, y.u, y_2.u


		self.y = y_1
		cpu = t0 - time.clock()
		return cpu, hashed_input

	
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
		for filename in glob.glob(self.casename+'_*.png'):
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

		self.plt.savefig(self.casename+'_%04d.png' % (n))

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
		for filename in glob.glob(self.casename+'_*.png'):
			os.rename(filename, os.path.join(directory, filename))
		os.chdir(directory)        # cd directory
		fps = 4 # frames per second
		if self.backend is not None:
			from scitools.std import movie
			movie(self.casename+'_*.png', encoder='html',
			      output_file='index.html', fps=fps)

		# Создаем другие видео форматы: Flash, Webm, Ogg, MP4
		codec2ext = dict(flv='flv', libx264='mp4', libvpx='webm',
		                 libtheora='ogg')
		filespec = self.casename+'_%04d.png'
		movie_program = 'ffmpeg' #'avconv' # или 'ffmpeg' 
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

def experiment_with_solver():
	p = WaveProblem1D()
	l = float(p.prm['L'][1])
	p.prm['c'] = 0.5
	p.prm['T'] = 4
	p.prm['I'] = lambda x: np.where(np.abs(x-l/2.) > 0.1,0.,1.)
	p.prm['ul'] = None
	p.prm['ur'] = None
	s = WaveExplicitSolver1D(p)
	s.prm['dt'] = (l/10)/p.prm['c']

	action = PlotAndStoreSolution(umax=1.1, umin=-1.1, backend=None, screen_movie=False, filename='test')
	s.prm['user_action'] = action
	cpu, hashed_input =s.solve()
	action.make_movie_file()
	action.close_file(hashed_input)
	
if __name__ == '__main__':
	experiment_with_solver()
