import numpy as np
from numpy import arange

a = np.arange(5)
print(a)
print(a.dtype)
print(a.shape)

m = np.array([arange(2),arange(2)])
print(m)
print(m.shape)

n = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(n)
print(n.shape)
print(n[1,:])

print(n.dtype.itemsize)

b = np.arange(24).reshape(2,3,4)
print(b)
print(b.shape)
print(b[:,0,0])
print(b[1])
print(b[1,1])
print(b[0,...])
print(b[:,1])

print(b.ravel())
print(b.flatten())

b.shape=(4,6)
print(b)

print(b.transpose())
print(b)

print(b.resize((3,8)))



c = np.arange(9).reshape(3,3)
d = 2*c
print(np.hstack((c,d)))
print(np.concatenate((c,d), axis=1))


print(np.hsplit(c,3))
print(np.split(c,3,axis=1))


i2 =np.eye(2)
print(i2)
np.savetxt("eye.txt",i2)

