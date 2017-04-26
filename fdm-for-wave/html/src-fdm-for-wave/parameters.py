# -*- coding: utf-8 -*-
class Parameters(object):
	def __init__(self):
		"""
		Подклассы должны инициализировать self.prm параметрами и значениями
		по умолчанию, self.type соответствующими типами, и self.help 
		соответствующими описаниями параметров. self.type и self.help являются 
		необязательными. self.prm должен быть заполнен и содержать все 
		параметры
		"""
		pass

	def ok(self):
		""" Проверяет определены ли атрибуты класса prm, type и help """
		if hasattr(self, 'prm') and isinstance(self.prm, dict) and \
		   hasattr(self, 'type') and isinstance(self.type, dict) and \
		   hasattr(self, 'help') and isinstance(self.help, dict):
			return True
		else:
			raise ValueError(
				'Конструктор в классе %s  не инициализирует\n'\
				'словари self.prm, self.type, self.help!' %
				self.__class__.__name__)

	def _illegal_parameter(self, name):
		"""Вызывает исключение о недопустимом имени параметра """
		raise ValueError(
			'Параметр "%s" не зарегистрирован.\n' \
			'Допустимые параметры: \n%s' %
			(name, ' '.join(list(self.prm.keys()))))

	def set(self, **parameters):
		""" Устанавливаем значения один или несколько параметров. """
		for name in parameters:
			if name in self.prm:
				self.prm[name] = parameters[name]
			else:
				slef._illegal_parameter(name)

	def get(self, name):
		""" Возвращает значения одного или нескольких параметров. """
		if isinstance(name, (list, tuple)):
			for n in name:
				if n not in self.prm:
					self._illegal_parameter(n)
			return [self.prm[n] for n in name]
		else:
			if name not in self.prm:
				self._illegal_parameter(name)
			return self.prm[name]

	def __getitem__(self, name):
		""" Разрешает доступ к параметру по obj[name] """
		return self.get(name)

	def __setitem__(self, name, value):
		""" Допускает синтаксис obj[name] для задания значения параметра """
		return self.set(name=value)

	def define_command_line_options(self, parser=None):
		self.ok()
		if parser is None:
			import argparse
			parser = argparse.ArgumentParser()

		for name in self.prm:
			tp = self.type[name] if name in self.type else str
			help = self.help[name] if name in self.help else None
			parser.add_argument(
				'--' + name, default=self.get(name), metavar=name,
				type=tp, help=help)

		return parser


