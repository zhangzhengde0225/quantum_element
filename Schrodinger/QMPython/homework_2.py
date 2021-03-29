"""
在homework1的基础上，求绝对值、复共轭积分.
$$
\psi = A*e^{-\lambda \abs{x} - i\omega t}
$$
"""

from sympy import *
from sympy.functions import Abs, conjugate
import matplotlib.pyplot as plt
import numpy as np


def psi_f(x, sig):
	return np.sqrt(1 / (np.sqrt(2) * sig)) * np.exp(-np.abs(x) / (np.sqrt(2) * sig))


def run():
	A, lam, x, om, t = symbols("A lam x om t", real=True)
	assumptions.assume.global_assumptions.add(Q.positive(lam))
	psi = Function('psi')(x)
	psi = A * exp(-lam * Abs(x) - I * om * t)
	ret = simplify(conjugate(psi) * psi)  # 简化
	print(ret)  # A**2*exp(-2*lam*Abs(x))

	ret = integrate(conjugate(psi) * psi, (x, -oo, oo))
	print(ret)  # (A**2/lam, Abs(arg(lam)) < pi/2)
	# 所以求得系数 A = sqrt(lam)

	psi = sqrt(lam) * exp(-lam * Abs(x) - I * om * t)

	# 已知波函数了，求解<x>和<x**2>
	print(integrate(conjugate(psi) * x * psi, (x, -oo, oo)))  # 0
	print(integrate(conjugate(psi) * x * x * psi, (x, -oo, oo)))  #


def run2():
	"""
	令t=0, 画不同的sigma对应的波函数
	:return:
	"""
	x_r = np.arange(-5, 5, 0.001)
	x_ave = 0
	for s in [0.5, 1, 2]:
		p = plt.plot(x_r, psi_f(x_r, s), label=f'sigma={s}')
		x1 = x_ave-s
		plt.plot([x1, x1], [-0.05, 0.1], color=p[0].get_color(), linestyle='-', linewidth=3)
		x2 = x_ave+s
		plt.plot([x2, x2], [-0.05, 0.1], color=p[0].get_color(), linestyle='-', linewidth=3)

	plt.legend()
	plt.title(f'Graph of the wavefunction for t=0')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True, which='both')
	plt.show()


if __name__ == '__main__':
	run()
	run2()
