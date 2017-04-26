# -*- coding: utf-8 -*-
import numpy as np

class Mesh(object):
	"""
	Содержит структуру данных для равномерной сетки на гиперкубе в 
	пространстве и равномерной сетки по времени.

	======== ===================================================
	Параметр         Описание
	======== ===================================================
	L         Список двухэлементных списков с минимальной и 
	          максимальной координатами по каждому 
	          пространственному направлению
	T         Конечный момент времени
	Nt        Число отрезков по времени
	dt        Шаг по времени. Или Nt или dt должно быть задано
	N         Список, содержащий количество отрезков
	          по каждому пространственному направлению.
	d         Список шагов по пространственным переменным.
	          Или N или d должен быть задан.
	======== ===================================================

	Пользователь имеет доступ ко всем описанным выше параметрам плюс 
	к ``x[i]`` и ``t`` для координат в направлении ``i`` и временной 
	координаты, соответственно.

	Примеры:

	>>> from UniformFDMesh import Mesh
	>>>
	>>> # Простая пространственная сетка
	>>> m = Mesh(L=[0,1], N=4)
	>>> print m.dump()
	пространство: [0,1] N = 4 d = 0.25
	>>>
	>>> # Простая сетка по времени
	>>> m = Mesh(T=4, dt=0.5)
	>>> print m.dump()
	время: [0,4] Nt = 8 dt = 0.5
	>>>
	>>> # двумерная пространственная сетка
	>>> m = Mesh(L=[[0,1], [-1,1]], d=[0.5, 1])
	>>> print m.dump()
	пространство: [0,1]x[-1,1] N = 2x2 d = 0.5,1
	>>>
	>>> # двумерная пространственная сетка и сетка по времени
	>>> m = Mesh(L=[[0,1], [-1,1]], d=[0.5, 1], Nt=10, T=3)
	>>> print m.dump()
	пространство: [0,1]x[-1,1] N = 2x2 d = 0.5,1 время: [0,3] Nt = 10 dt = 0.3

	"""

	def __init__(self,
	             L=None, T=None, t0=0,
	             N=None, d=None,
	             Nt=None, dt=None):
		if N is None and d is None:
			# Пространственная сетка отсутсвует
			if Nt is None and dt is None:
				raise ValueError(
					'Коструктор сетки: Должен быть задан либо Nt, либо dt')
			if T is None:
				raise ValueError(
					'Конструктор сетки: T должен быть задан')
		if Nt is None and dt is None:
			# Отсутствует сетка по времени
			if N is None and d is None:
				raise ValueError(
					'Конструктор сетки: Должен быть задан либо N, либо d')
			if L is None:
				raise ValueError('Конструктор сетки: L должен быть задан')
		
		# Допускаем одномерный интерфейс без использования вложенных списков
		if L is not None and isinstance(L[0], (float,int)):
			L=[L]
		if N is not None and isinstance(N, (float,int)):
			N=[N]
		if d is not None and isinstance(d, (float,int)):
			d=[d]

		# Устанавливаем все атрибуты в None
		self.x = None
		self.t = None
		self.T = None
		self.Nt = None
		self.dt = None
		self.L = None
		self.N = None
		self.d = None
		self.t0 = t0

		if N is None and d is not None and L is not None:
			self.L = L
			if len(d) != len(L):
				raise ValueError('список d имеет длину отличную от L: %d - %d',
				                 len(d), len(L))
			self.d = d
			self.N = [int(round(float(self.L[i][1] - self.L[i][0])/d[i]))
			          for i in range(len(d))]
		if d is None and N is not None and L is not None:
			self.L = L
			if len(N) != len(L):
				raise ValueError('список N имеет длину отличную от L: %d - %d',
				                 len(N), len(L))
			self.N = N
			self.d = [float(self.L[i][1] - self.L[i][0])/N[i]
			          for i in range(len(N))]

		if Nt is None and dt is not None and T is not None:
			self.T = T
			self.dt = dt
			self.Nt = int(round(T/dt))
		if dt is None and Nt is not None and T is not None:
			self.T = T
			self.Nt = Nt
			self.dt = T/float(Nt)

		if self.N is not None:
			self.x = [np.linspace(self.L[i][0], self.L[i][1], self.N[i]+1)
			          for i in range(len(self.L))]
		if self.Nt is not None:
			self.t = np.linspace(self.t0, self.T, self.Nt+1)

	def get_num_space_dim(self):
		return len(self.d) if self.d is not None else 0

	def has_space(self):
		return self.d is not None

	def has_time(self):
		return self.dt is not None

	def dump(self):
		s = ''
		if self.has_space():
			s += 'пространство: ' + \
			     'x'.join(['[%g,%g]' % (self.L[i][0],self.L[i][1])
			               for i in range(len(self.L))]) + ' N = '
			s += 'x'.join([str(Ni) for Ni in self.N]) + ' d = '
			s += ','.join([str(di) for di in self.d])
		if self.has_space() and self.has_time():
			s+= ' '
		if self.has_time():
			s += 'время: ' + '[%g,%g]' % (self.t0, self.T) + \
			     ' Nt = %g' % self.Nt + ' dt = %g' % self.dt
		return s

class Function(object):
	"""
	Скалярная или векторная сеточная функция (используется класс Mesh)
	=========== ==============================================================
	Параметр       Описание
	=========== ==============================================================
	mesh         объект класса Mesh: пространственная и/или временная сетка
	num_comp     Количество компонент функции (1 для скалярной функции)
	space_only   True, если функция определена только на пространственной
	             сетке. False, если функция зависит от пространства и времени
	=========== ==============================================================

	Индексация массива ``u``, который содержит значения функции в узлах сетки,
	зависит от того, есть ли у нас пространственная и/или временная сетка.

	Примеры:
	
	>>> from UniformFDMesh import Mesh, Function
	>>>
	>>> # Простая пространственная сетка
	>>> m = Mesh(L=[0,1], N=4)
	>>> print m.dump()
	пространство: [0,1] N = 4 d = 0.25
	>>> f = Function(m)
	>>> f.indices
	['x0']
	>>> f.u.shape
	(5,)
	>>> f.u[4]  # значение в узле с номером 4
	0.0
	>>>
	>>> # Прострая сетка по времени для двух компонент
	>>> m = Mesh(T=4, dt=0.5)
	>>> print m.dump()
	время: [0,4] Nt = 8 dt = 0.5
	>>> f = Function(m, num_comp=2)
	>>> f.indices
	['time', 'component']
	>>> f.u.shape
	(9, 2)
	>>> f.u[3,1]  # значение на 3 временном слое, comp=1 (2-я компонента)
	0.0
	>>>
	>>> # двумерная пространственная сетка
	>>> m = Mesh(L=[[0,1], [-1,1]], d=[0.5, 1])
	>>> print m.dump()
	пространство: [0,1]x[-1,1] N = 2x2 d = 0.5,1
	>>> f = Function(m)
	>>> f.indices
	['x0', 'x1']
	>>> f.u.shape
	(3, 3)
	>>> f.u[1,2]  # значение в узле (1,2)
	0.0
	>>>
	>>> # двумерная пространственная сетка и сетка по времени
	>>> m = Mesh(L=[[0,1],[-1,1]], d=[0.5,1], Nt=10, T=3)
	>>> print m.dump()
	пространство: [0,1]x[-1,1] N = 2x2 d = 0.5,1 время: [0,3] Nt = 10 dt = 0.3
	>>> f = Function(m, num_comp=2, space_only=False)
	>>> f.indices
	['time', 'x0', 'x1', 'component']
	>>> f.u.shape
	(11, 3, 3, 2)
	>>> f.u[2,1,2,0]  # значение на 2-м временном слое, в пространственном узле (1,2), comp=0
	0.0
	>>> # Функция с данными только по вространству
	>>> f = Function(m, num_comp=1, space_only=True)
	>>> f.indices
	['x0', 'x1']
	>>> f.u.shape
	(3, 3)
	>>> f.u[1,2]  # значение в узле (1,2)
	0.0
	"""

	def __init__(self, mesh, num_comp=1, space_only=True):
		self.mesh = mesh
		self.num_comp = num_comp
		self.space_only = space_only
		self.indices = []

		# Создаем массив(ы) для хранения значений в узлах сетки
		if (self.mesh.has_space() and not self.mesh.has_time()) or \
		   (self.mesh.has_space() and self.mesh.has_time() and space_only):
			# Только пространственная сетка
			if num_comp == 1:
				self.u = np.zeros([self.mesh.N[i]+1 for i in range(len(self.mesh.N))])
				self.indices = ['x'+str(i) for i in range(len(self.mesh.N))]
			else:
				self.u = np.zeros([self.mesh.N[i]+1 for i in range(len(self.mesh.N))] + [num_comp])
				self.indices = ['x'+str(i) for i in range(len(self.mesh.N))] + ['component']
		if not self.mesh.has_space() and self.mesh.has_time():
			# Только сетка по времени
			if num_comp == 1:
				self.u = np.zeros(self.mesh.Nt + 1)
				self.indices = ['time']
			else:
				self.u = np.zeros((self.mesh.Nt+1, num_comp))
				self.indices = ['time', 'component']
		if self.mesh.has_space() and self.mesh.has_time() and not space_only:
			# Пространственно-временная сетка
			size = [self.mesh.Nt + 1] + [self.mesh.N[i]+1 for i in range(len(self.mesh.N))]
			if num_comp > 1:
				self.indices = ['time'] + \
				               ['x' + str(i) for i in range(len(self.mesh.N))] + \
				               ['component']
				size += [num_comp]
			else:
				self.indices = ['time'] + \
				               ['x' + str(i) for i in range(len(self.mesh.N))]
			self.u = np.zeros(size)
			

	def norm(self):
		return np.sqrt(self.u*self.u)

if __name__ =='__main__':
	import doctest
	failure_count, test_count = doctest.testmod()
