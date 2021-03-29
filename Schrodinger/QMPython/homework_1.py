"""
使用sympy求解积分
$$I =\int_{-\infty}^{\infty} A e^{-\lambda\left(x-a\right)^2}dx$$
"""

from sympy import *
import matplotlib.pyplot as plt
import numpy as np


def rho_f(x, a, sig):
	return np.sqrt(1/(2*np.pi*sig**2))*np.exp(-(x-a)**2/(2*sig**2))


def run():
	"""
	求解积分
	:return:
	"""

	# 定义符号
	A, lam, a, x = symbols("A lam a x", real=True)
	# 定义lam范围，为正
	assumptions.assume.global_assumptions.add(Q.positive(lam))

	rho = Function('rho')(x)
	rho = A*exp(-lam*(x-a)**2)

	ret = integrate(rho, (x, -oo, oo))
	print(ret)

	rho = sqrt(lam//pi)*exp(-lam*(x-a)**2)
	print(integrate(rho, (x, -oo, oo)))
	print(integrate(x*rho, (x, -oo, oo)))
	print(integrate(x*x*rho, (x, -oo, oo)))
	# 已解出


def run2():
	"""
	已求解积分画图
	:return:
	"""
	x_r = np.arange(-2, 10, 0.01)
	x_ave = 4
	for s in [0.5, 1, 2]:
		p = plt.plot(x_r, rho_f(x_r, x_ave, s), label=f'sigma={s}')
		x1 = x_ave - s
		plt.plot([x1, x1], [-0.05, 0.1], color=p[0].get_color(), linestyle='-', linewidth=3)
		x2 = x_ave + s
		plt.plot([x2, x2], [-0.05, 0.1], color=p[0].get_color(), linestyle='-', linewidth=3)


	plt.legend()
	plt.title('Normalize Gaussian with a=4')
	plt.xlabel('x')
	plt.ylabel('y')
	plt.grid(True, which='both')
	plt.show()


if __name__ == '__main__':
	run()
	run2()
