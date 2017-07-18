'''
3.7 无穷大与NaN
你想创建或测试正无穷、负无穷或 NaN(非数字)的浮点数
'''
# float() 创建这些特殊的浮点值
a = float('inf')
b = float('-inf')
c = float('nan')
print(a)  # inf
print(b)  # -inf
print(c)  # nan

import math
print(math.isinf(a))  #True
print(math.isnan(c))  #True

a = float('inf')
print(a + 45)  #inf
print(10/a)  #0.0

# 有些操作返回NaN结果
print(a/a)  #nan
b = float('-inf')
print(a+b)  #nan

# nan会在所有操作中传播，并不会产生异常
c = float('nan')
print(c*2)  # nan
print(math.sqrt(c))  #nna

# nan值一个特别的地方是 它们的比较操作总是返回False
d = float('nan')
print(c == d)  # False
print(c is d)  # False



'''
3.8 分数运算
'''
# fractions module
from fractions import Fraction
a = Fraction(5, 4)
b = Fraction(7, 16)
print(a + b)  #27/16
print(a * b)  #35/64

c = a*b
print(c.numerator)  #35
print(c.denominator) # 64
print(float(c))  #0.546875
print(c.limit_denominator(8)) #4/7

# Converting a float to a fraction
x = 3.75
y = Fraction(*x.as_integer_ratio())
print(y)  #15/4












