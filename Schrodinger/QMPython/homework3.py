"""
数值法求解波函数随时间的变化
"""

import numpy as np
import matplotlib.pyplot as plt


def E(n, a):
	return n * n * np.pi * np.pi / (2. * a * a)  # We set m/hbar =1


def psi_f(x, t, a):
	norm = np.sqrt(2/a)
	n_r = np.array(range(1, 100))
	phase = np.exp(-1.j*E(n_r, a)*t)
	out = c(n_r) * norm * np.sin(x*n_r*np.pi/a)*phase
	s = np.sum(out)
	return s


def c(n):
	assert type(n) == np.ndarray, f'{type(n)}'

	cn = 4 * np.sqrt(6) / (n * n * np.pi * np.pi)
	# print(cn)
	cn[n % 2 == 0] = 0
	# print(cn, n)
	return cn


def plot_real():
	# 画实部
	x_r = np.arange(0, 10, 0.01)
	for t in np.arange(0., 65., 8.):
		y_p = [np.real(psi_f(x, t, 10.)) for x in x_r]
		plt.plot(x_r, y_p, label=f't={t}')
	plt.legend()
	plt.title('real part of Wave Function')
	plt.grid(True, which='both')
	plt.show()


def plot_imagenary():
	x_r = np.arange(0, 10, 0.01)
	for t in np.arange(0., 65., 8.):
		y_p = [np.imag(psi_f(x, t, 10.)) for x in x_r]
		plt.plot(x_r, y_p, label="t={}".format(t))
		plt.legend()
		plt.title("Imaginary part of Wave Function")
		plt.xlabel("x")
		plt.ylabel("y")
		plt.grid(True, which='both')
	plt.show()


def plot_abs():
	x_r = np.arange(0, 10, 0.1)
	for t in np.arange(0., 64., 4.):
		y_p = [np.abs(psi_f(x, t, 10.)) for x in x_r]
		plt.plot(x_r, y_p, label="t={}".format(t))
	plt.legend()
	plt.title("Absolute part of Wave Function")
	# plt.xlabel("x")
	# plt.ylabel("y")
	plt.grid(True, which='both')
	plt.show()


def run():
	nl = np.array(range(1, 70000))
	a = c(nl)
	print(a)
	sum = np.sum(c(nl) ** 2)
	print(sum)

	# 画出实部
	# plot_real()
	# plot_imagenary()
	plot_abs()



if __name__ == '__main__':
	run()
