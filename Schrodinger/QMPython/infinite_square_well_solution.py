import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt


def f(x, a):
	x = np.array([x]) if isinstance(x, int) else x
	x = np.array([x]) if isinstance(x, float) else x
	x = np.array(x) if isinstance(x, list) else x
	x = x.astype(np.float)
	# print(type(x))
	assert type(x) == np.ndarray
	norm = np.sqrt(12. / (a * a * a))
	return np.piecewise(x, [x < a / 2., x >= a / 2.], [lambda x: norm * x, lambda x: norm * (a - x)])


def psi_n(x, n, a):
	return np.sqrt(2. / a) * np.sin(n * x * np.pi / a)


def init_fun(x, n, a):
	return f(x, a) * psi_n(x, n, a)


def c(n, a):
	if n == 0 or n % 2 == 0:
		return
	return spi.quad(init_fun, 0, a, args=(n, a), limit=100)[0]


def run():
	# print(f_slow(0, 10.), f_slow(3, 10), f_slow(5, 10), f_slow(7, 10), f_slow(10, 10))

	a = np.array([3.])
	# a = 3.
	# print(f(a, 10.))
	# print(f(np.array([0, 3, 5, 7, 10], dtype=float), 10))
	# exit()

	"""
	# fig0 = plt.figure(figsize=(10, 8))
	x_r = np.arange(0, 10, 0.01)
	plt.plot(x_r, f(x_r, 10))
	plt.show()
	"""

	Nmax = 40
	a_l = 10.
	a_step = 10./100
	nl = np.array(range(Nmax))
	cx = np.array([c(n, a_l) for n in nl])
	print(cx)
	print(cx.shape)
	print(cx[0], cx[1], cx[2], cx[3], cx[4])
	print(np.sum(cx * cx), cx[Nmax - 1])


if __name__ == '__main__':
	run()
