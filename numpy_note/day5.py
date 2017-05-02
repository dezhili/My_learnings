# array合并

import numpy as np 

A = np.array([1,1,1])
B = np.array([2,2,2])
C = np.vstack((A,B))
print(C)  #上下合并
print(C.T)
print(A.shape,C.shape)  #(3,) (2, 3) 第一个含三个元素的数组，C2行3列的矩阵


D = np.hstack((A,B))
print(D)  #左右合并
print(A.shape,D.shape)

print(A.T)  #[1 1 1] A数组
print(A[np.newaxis,:])  #[[1 1 1]]
print(A[np.newaxis,:].shape)
print(A[:,np.newaxis])
print(A[:,np.newaxis].shape)

E = np.array([1,1,1])[:,np.newaxis]  #转置
F = np.array([2,2,2])[:,np.newaxis]
print(E)

G = np.vstack((E,F))
H = np.hstack((E,F))
print(H)
print(E.shape,H.shape)

J = np.concatenate((E,F,F,E),axis=0)
K = np.concatenate((E,F,F,E),axis=1)
print(J)
print(K)






