# copy && deep copy
#在numpy中，a=b，a改变的话，b也会随之改变 和在python中不同，a就是b
import numpy as np 
a = np.arange(4)

b=a  #其实就是浅复制
c=b

a[1]=11
print(a)
print(b)
print(a is b)
print(c is a)

d = a.copy() #deep copy,b并没有关联a
a[3]=44
print(a)
print(d)
print(a is d)