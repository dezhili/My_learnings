#numpy 的基础运算
import numpy as np 
a = np.array([10,20,30,40])
b = np.arange(4)

#print(a)

# print(a,b)
# c = a**2
# print(c)

# d = 10*np.tan(a)
# print(d)

# print(b)
# print(b==3)

c = np.array([[1,1],
	          [0,1]])
d = np.arange(4).reshape((2,2))
e = c*d

# print(c.shape)

# print(d)

# print(e)

# e_dot = np.dot(c,d)
# e_dot_2 = c.dot(d)
# print(e_dot)
# print(e_dot_2)


f = np.random.random((2,4))
# print(f)

# print(np.sum(f,axis=1))  #对行求
# print(np.min(f,axis=0))  #对列求
# print(np.max(f))

A = np.arange(2,14).reshape((3,4))

print(A)
print(np.argmin(A)) #求最小值的索引
print(np.argmax(A))
print(np.mean(A))
print(A.mean())
print(np.average(A))
print(np.median(A))  #中位数
print(np.cumsum(A))  #累加 [ 2  5  9 14 20 27 35 44 54 65 77 90]
print(np.diff(A))  #累差
print(np.nonzero(A)) #非零(array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2], dtype=int64), array([0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], dtype=int64))

B = np.arange(14,2,-1).reshape((3,4))
print(np.sort(B))
print(np.transpose(B))  #转置
print((B.T).dot(B))
print(np.dot(B.T,B))

print(np.clip(B,5,9))  

print(np.mean(B,axis=0))



