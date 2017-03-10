# -*- coding: utf-8 -*-

import sympy as sym
V, t, U, omega, tau = sym.symbols('V t U omega tau')  # глобальные символы
f = None  # глобальная переменная для функции источника ОДУ

def ode_source_term(u):
    """
    Возвращает функцию источника ОДУ, равную u'' + omega**2*u.
    u --- символьная функция от t."""
    return sym.diff(u(t), t, t) + omega**2*u(t)

def residual_discrete_eq(u):
    """
    Возвращает невязку разностного уравнения на заданной u.
    """
    R = ...
    return sym.simplify(R)

def residual_discrete_eq_step1(u):
    """
    Возвращает невязку разностного уравнения на первом шаге 
    на заданной u.
    """
    R = ...
    return sym.simplify(R)

def DtDt(u, tau):
	"""
	Возвращает вторую разностную производную от u.
	u --- символьная функция от t.
    """
    return ...

def main(u):
	"""
	Задавая некоторое решение u как функцию от t, используйте метод
	пробных функций для вычисления функции источника f и проверьте 
	является ли u решением и разностной задачи.
    """
    print '=== Проверка точного решения: %s ===' % u
    print "Начальные условия u(0)=%s, u'(0)=%s:" % \
          (u(t).subs(t, 0), sym.diff(u(t), t).subs(t, 0))

    # Метод пробных функций требует подбора f
    global f
    f = sym.simplify(ode_lhs(u))

    # Невязка разностной задачи (должна быть 0)
    print 'residual step1:', residual_discrete_eq_step1(u)
    print 'residual:', residual_discrete_eq(u)

def linear():
    main(lambda t: V*t + U)

if __name__ == '__main__':
    linear()
