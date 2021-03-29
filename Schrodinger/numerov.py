"""
打靶法和numerov方法求解一维无限深势阱和抛物线形势的运动粒子的波函数，
失败：https://zhuanlan.zhihu.com/p/59099100?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0LU8IYt1
"""
import numpy as np


def psi_normlize(psi):
	"""传入波函数，归一化: 模方的和为)"""
	norm = np.power(psi, 2)  # 模方
	norm_normed = norm/np.sum(norm)
	psi = np.sqrt(norm_normed)
	# print(np.sum(norm_normed))
	# print(np.sum(psi))
	# print(psi)
	return psi


def cal_psi2(psi1, psi0, h, k):
	psi2 = (2 * (12 - 5 * h * h * k * k))/(12 + h * h * k * k)*psi1 - psi0
	return psi2


def solve(x, L=1):
	s = x/L
	m = 500
	zeta = 10
	e = 0.1
	# 边界条件
	psi_s1 = 0
	crit = 1e-10

	k = 2 * e
	h = L / m  # 0.002

	# 初始值
	psi0 = 0
	psi1 = 0.1


	psi = np.zeros(m)
	while True:

		for i in range(m):
			psi2 = cal_psi2(psi1=psi1, psi0=psi0, h=(i+1)*h, k=k)
			psi0 = psi1
			psi1 = psi2
			psi[i] = psi2
			print(f'{i} {psi2:.4f}')

		psi = psi_normlize(psi)  # 波函数模方和为一
		# 要使得这个由初始k计算出的波函数满足第二个边界条件psi(s=1)=0
		loss = psi[-1] - psi_s1
		e = k/2

		print(f'current energy: {e:.6f}, loss: {loss:.8f}, k: {k:>2f}') if loss < 1e-4 else None
		if loss <= crit:
			break
		k = k - 0.0001


if __name__ == '__main__':
	solve(x=1)
