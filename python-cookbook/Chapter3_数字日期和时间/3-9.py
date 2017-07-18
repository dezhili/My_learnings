'''
3.9 大型数组运算
'''
# solutions: numpy

# python lists
x = [1, 2, 3, 4]
y = [5, 6, 7, 8]
print(x * 2)  #[1, 2, 3, 4, 1, 2, 3, 4]
print(x + y)  #[1, 2, 3, 4, 5, 6, 7, 8]

# numpy arrays
import numpy as np
ax = np.array([1, 2, 3, 4])
ay = np.array([5, 6, 7, 8])
print(ax * 2)  #[2 4 6 8]
print(ax + 10)  #[11 12 13 14]
print(ax * ay)  #[ 5 12 21 32]

# numpy 还为数组操作提供了大量通用的函数, 作为math 中类似函数的替代
print(np.sqrt(ax))  #[ 1.          1.41421356  1.73205081  2.        ]
print(np.cos(ax))  #[ 0.54030231 -0.41614684 -0.9899925  -0.65364362]

# NumPy 数组使用了C或者Fortran语言的机制分配内存。
# 也就是说，它们是一个非常大的连续的并由同类型数据组成的内存区域。
# 所以，你可以构造一个比普通Python列表大的多的数组。
grid = np.zeros(shape=(10000, 10000), dtype=float)
print(grid)

# numpy 扩展Python列表的索引功能。特别是多维数组
a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a.shape, a.ndim)
# select row 1
print(a[1])  # [5 6 7 8]
# select column 1
print(a[:, 1])  #[ 2  6 10]
print(a[1:3, 1:3])
a[1:3, 1:3] += 10
print(a)

# Broadcast a row vector across an operation on all rows
print(a + [100, 101, 102, 103])

# Condtional assignment on an array
print(np.where(a<10, a, 10))
print('-------------------------------------------------------------------------------')

'''
3.10 矩阵与线性代数运算
执行矩阵和线性代数运算，比如矩阵乘法、寻找行列式、求解线性方程组
'''
# Numpy
import numpy as np
m = np.matrix([[1, -2, 3], [0, 4, 5], [7, 8, -9]])
print(m)
print(m.T)
print(m.I)  # Return inverse

v = np.matrix([[2], [3], [4]])
print(m * v)

# np.linalg 子包中找到更多的操作函数
import numpy.linalg

print(numpy.linalg.det(m))  # 行列式
print(numpy.linalg.eigvals(m))  # 特征值
print(numpy.linalg.solve(m, v))  # Solve for x in mx=v





