#array 分割

import numpy as np 

A = np.arange(12).reshape((3,4))
print(A)

#纵向分割
print(np.split(A,2,axis=1)) #分成2块,相等的分割

#横向分割
print(np.split(A,3,axis=0)) #分成3块

#实现不等量的分割
print(np.array_split(A,3,axis=1))

print(np.vsplit(A,3))  #横向分割
print(np.hsplit(A,2))  #纵向分割