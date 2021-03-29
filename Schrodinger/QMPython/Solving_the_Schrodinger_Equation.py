"""
离散化数值方法求解薛定谔方程
"""

import numpy as np
import matplotlib.pyplot as plt


def hamiltonian_matrix(m, N, step, V):
	"""
	the matrization of Hamiltonian
	$$
	\hbar=\frac{\bbar}{2m}\frac{d^2}{dx^2} + V
	$$
	:return:
	"""
	hbar = 6.62607015e-34 / np.pi
	hbar = 1
	sodm = np.diag(np.ones(N) * -2, 0) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
	sodm = - sodm * hbar * hbar / (2 * m * step * step)
	assert sodm.shape == V.shape
	H = sodm + V
	return H


def run():
	N = 128
	a = np.pi
	x = np.linspace(0, 2*np.pi, N)

	h = x[1] - x[0]  # 步长
	y = np.sin(x)

	# first order derivative matrix
	fodm = np.diag(-1*np.ones(N), 0) + np.diag(np.ones(N-1), 1)
	fodm = fodm/h

	yp = fodm.dot(y)  # 一阶导数
	print(fodm, fodm.shape, y.shape, yp.shape, i)

	# second order derivative matrix
	sodm = np.diag(np.ones(N)*-2, 0) + np.diag(np.ones(N-1), 1) + np.diag(np.ones(N-1), -1)
	sodm = sodm/(h*h)

	ypp = sodm.dot(y)

	print(ypp, ypp.shape)

	# plt.figure(figsize=(10, 7))
	plt.plot(x, y)
	plt.plot(x[:-1], yp[:-1])
	plt.plot(x[1:-1], ypp[1:-1])
	plt.show()


def run2():
	m = 1
	a = 1.0
	N = 512

	x = np.linspace(-a/2, a/2, N)
	step = x[1] - x[0]

	V = np.zeros_like(x)
	V = np.ones_like(x)

	H = hamiltonian_matrix(m, N, step, np.diag(V))
	E, psiT = np.linalg.eigh(H)
	psi = np.transpose(psiT)

	print(E.shape, psi.shape)  # (N, ) (N, N)
	print(psi)

	for i in range(5):
		# print(psi[i], psi[i].shape, psi[i][N-10])
		# exit()
		if psi[i][N-10] < 0:
			plt.plot(x, -psi[i] / np.sqrt(step), label="$E_{}$={:>8.3f}".format(i, E[i]))
		else:
			plt.plot(x, psi[i] / np.sqrt(step), label="$E_{}$={:>8.3f}".format(i, E[i]))
		plt.title("Solutions to the Infinite Square Well")

	plt.legend()
	# plt.savefig("Infinite_Square_Well_WaveFunctions.pdf")
	# plt.show()

	sum = np.sum(psi/np.sqrt(step), axis=0)
	plt.plot(x, sum)
	print(sum.shape, sum)
	plt.show()


if __name__ == '__main__':
	# run()
	run2()
