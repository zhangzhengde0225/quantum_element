"""
华师大的实现
https://blog.csdn.net/weixin_42549154/article/details/111962351
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib



def a():
	x = np.linspace(-2, 2, 100)
	f = np.sin(x) / x
	print(x.shape, f.shape)
	plt.plot(x, f)
	plt.title('xx')
	plt.show()


if __name__ == '__main__':
	a()
