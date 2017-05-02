#索引

import numpy as np 
A = np.arange(8)
# print(A)
# print(A[3])

B = np.arange(12).reshape((3,4))
#print(B)
print(B[1])
print(B[2][1])
print(B[1:])


print(B[1][:2])  #[4 5]
print(B[1,:2])  #[4 5]

print(B[1:,:2])

for row in B:
	print(row) 

for column in B.T:
	print(column)

C = np.arange(3,15).reshape((3,4))
print(C.flatten())   #[ 3  4  5  6  7  8  9 10 11 12 13 14]

for item in C.flat:
	print(item)  #按列展开

