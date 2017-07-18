'''
3.1 数字的四舍五入
你想对浮点数执行指定精度的舍入运算
'''
# round(value, ndigits)
print(round(1.23, 1))  #1.2
print(round(-1.27, 1))  #-1.3
print(round(1.25363, 3))  #1.254

# 当一个值刚好在两个边界的中间的时候，round()返回离它最近的偶数
print(round(1.5))  #2
print(round(2.5))  #2

# 当传给round()的ndigits参数为负数的时候，舍入运算会作用在十位、百位、千位
print(round(1627731, -1))  #1627730
print(round(1627731, -2))  #1627700
print(round(1627731, -3))  #1627800

# 舍入和格式化输出不一样。如果只是简单的输出一定宽度的数，不需要使用round()
x = 1.23456
print(format(x, '0.2f'))  #1.23
print(format(x, '0.3f'))  #1.235
print('value is {:0.3f}'.format(x))  #value is 1.235


'''
3.2 执行精确的浮点数运算
'''
# 浮点数的一个普遍问题是它们并不能精确地表示十进制数。
a = 4.2
b = 2.1
print(a+b)  #6.300000000000001
print((a+b) == 6.3)  #False

# 如果想更精确(并能容忍一定的性能损耗), 使用decimal模块
from decimal import Decimal
a = Decimal('4.2')
b = Decimal('2.1')
print(a+b)  #6.3
print((a+b) == Decimal('6.3'))  #True

# decimal 模块的一个主要特征是允许你控制计算的每一方面，包括数字位数和四舍五入运算。
# 为了这样做，你先得创建一个本地上下文并更改它的设置
from decimal import localcontext
a = Decimal('1.3')
b = Decimal('1.7')
print(a/b)  #0.7647058823529411764705882353

with localcontext() as ctx:
    ctx.prec = 3
    print(a/b)  #0.765

nums = [1.23e+18, 1, -1.23e+18]
print(sum(nums))  #0.0   1 disappears

import math
print(math.fsum(nums))  #1.0


















