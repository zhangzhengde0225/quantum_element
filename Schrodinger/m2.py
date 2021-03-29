"""
一维方势阱中粒子能级和波函数matlab求解
https://blog.csdn.net/qq_33689250/article/details/103091661?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.control&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-5.control
"""

import numpy as np


def solve():
	ans = []
	for i in range(1, M-1):
		E = E + de
		r1 = np.sqrt(np.abs(E-V0))
		r2 = np.sqrt(np.abs(E))
		phi = (r2/r1)*np.sin(W*r1) + np.cos(W*r1)
		phi1 = r2*np.cos(W*r1) - r1*np.sin(W*r1)
		



if __name__ == '__main__':
	# 初始参数
	N = 0
	V0 = -20  # 势阱深度
	W = 1.0  # 势阱宽度
	Emin = V0  # 猜测能量最小值
	Emax = 0  # 猜测能量最大值
	M = 51  # 离散步长
	de = (Emax-Emin)/(M-1)
	E = Emin - de  # 为了便于循环
	ret = solve()








