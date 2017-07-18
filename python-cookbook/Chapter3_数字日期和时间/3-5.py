'''
3.5 字节到大整数的打包与解包
有一个字节字符串想将它解压成一个整数。 将一个大整数转换为一个字节字符串
'''
data = b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
print(len(data))  #16

#  将bytes解析为整数，int.from_bytes()
print(int.from_bytes(data, 'little'))  #69120565665751139577663547927094891008
print(int.from_bytes(data, 'big'))  #94522842520747284487117727783387188

# 将一个大整数转换为一个字节字符串, int.to_bytes(),并指定字节数和字节顺序
x = 94522842520747284487117727783387188
print(x.to_bytes(16, 'big'))  #b'\x00\x124V\x00x\x90\xab\x00\xcd\xef\x01\x00#\x004'
print(x.to_bytes(16, 'little'))  #b'4\x00#\x00\x01\xef\xcd\x00\xab\x90x\x00V4\x12\x00'

# 下面还有，并没有介绍


'''
3.6 复数的数学运算
你写的最新的网络认证方案代码遇到了一个难题，并且唯一的解决办法就是使用复数空间。
或者你仅仅需要使用复数执行一些计算操作
'''
# complex(real, imag) 或带有后缀的浮点数
a = complex(2, 4)
b = 3 - 5j
print(a)  #(2+4j)
print(b)  #(3-5j)
print(a.real)  # 2.0
print(a.imag)  # 4.0
print(a.conjugate())  # 共轭复数 (2-4j)

# 常见的数学运算都可以工作
print(a+b)
print(a*b)
print(abs(a))
print(a/b)

# 如果执行其他的复数函数比如 正弦 余弦 或平方根 使用 cmath module
import cmath
print(cmath.sin(a))
print(cmath.cos(a))
print(cmath.exp(a))

# 使用numpy，可以很容易的构造一个复数数组并执行各种操作
import numpy as np
a = np.array([2+3j, 4+5j, 6-7j, 8+9j])
print(a)  #[ 2.+3.j  4.+5.j  6.-7.j  8.+9.j]
print(a+2)  #[  4.+3.j   6.+5.j   8.-7.j  10.+9.j]
print(np.sin(a))

# Python的标准数学函数确实情况下并不能产生复数值，因此你的代码中不可能会出现复数返回值。
# 如果你想生成一个复数返回结果，你必须显示的使用 cmath 模块，或者在某个支持复数的库中声明复数类型的使用。
import cmath
print(cmath.sqrt(-1))  #1j


