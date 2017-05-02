#numpy 创建array
import numpy as np 

a = np.array([2,2,3],dtype=np.int)
print(a.dtype)
print(a)

a = np.array([[1,21,3],
	          [2,32,4],
	          [3,43,5]])
print(a)

b = np.zeros((3,4))
print(b)

c = np.ones((3,4),dtype = np.int16)
print(c)

d = np.empty((3,4))
print(d)

# e = np.arange(10,20,2) #类似range,生成有序的序列或矩阵
# print(e)

# f = np.arange(12).reshape((3,4))
# print(f)

# g = np.linspace(1,10,6).reshape((2,3))  #生成线段
# print(g)



